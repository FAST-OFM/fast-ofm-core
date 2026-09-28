// Exact exhaustive masked NMI. No NumPy ABI, camera access or runtime compiler.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {
struct Buffer {
    Py_buffer view{};
    ~Buffer() { if (view.obj) PyBuffer_Release(&view); }
    bool acquire(PyObject* object) {
        if (PyObject_GetBuffer(object, &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT))
            return false;
        if (view.ndim != 2 || view.itemsize != 1 || !view.format ||
            std::strcmp(view.format, "B") != 0) {
            PyErr_SetString(PyExc_ValueError, "NMI buffers must be contiguous uint8 planes");
            return false;
        }
        return true;
    }
};

struct Grid {
    int height, width, bins, maximum_x, maximum_y;
    double required_fraction;
};

// No C++ exception may cross either the released-GIL region or the Python ABI.
int calculate(const uint8_t* r, const uint8_t* g, const uint8_t* mask,
              const Grid& grid, std::vector<double>& scores) noexcept {
    try {
        const auto [h, w, bins, mx, my, fraction] = grid;
        const int pixels = h * w;
        int support = 0;
        for (int i = 0; i < pixels; ++i) {
            if (r[i] >= bins || g[i] >= bins) return 3;
            support += mask[i] != 0;
        }
        const double required = support * fraction;
        std::vector<double> lut(static_cast<size_t>(pixels) + 1, 0);
        for (size_t i = 1; i < lut.size(); ++i) lut[i] = i * std::log(double(i));
        std::vector<int> joint(bins * bins), rows(bins), cols(bins);
        scores.assign((2 * mx + 1) * (2 * my + 1),
                      std::numeric_limits<double>::quiet_NaN());
        size_t index = 0;
        for (int dy = -my; dy <= my; ++dy) {
            for (int dx = -mx; dx <= mx; ++dx, ++index) {
                std::fill(joint.begin(), joint.end(), 0);
                std::fill(rows.begin(), rows.end(), 0);
                std::fill(cols.begin(), cols.end(), 0);
                int n = 0;
                for (int y = std::max(0, -dy); y < std::min(h, h - dy); ++y) {
                    for (int x = std::max(0, -dx); x < std::min(w, w - dx); ++x) {
                        const int a = y * w + x, b = (y + dy) * w + x + dx;
                        if (!mask[a] || !mask[b]) continue;
                        // Recheck before indexing even if an external caller mutates a buffer.
                        const unsigned red = r[a], green = g[b];
                        if (red >= unsigned(bins) || green >= unsigned(bins)) return 3;
                        ++joint[red * bins + green];
                        ++rows[red]; ++cols[green]; ++n;
                    }
                }
                if (n < required) continue;
                // Preserve the original first-admissible-overlap refusal, not a skip.
                if (n < 16) return 1;
                double sr = 0, sg = 0, sj = 0;
                for (int i = 0; i < bins; ++i) { sr += lut[rows[i]]; sg += lut[cols[i]]; }
                for (int count : joint) sj += lut[count];
                const double logn = std::log(double(n));
                const double hr = logn - sr / n, hg = logn - sg / n, hj = logn - sj / n;
                if (!std::isfinite(hr + hg) || hr + hg <= 1e-12) return 2;
                scores[index] = std::clamp(2 * (hr + hg - hj) / (hr + hg), 0.0, 1.0);
            }
        }
        return 0;
    } catch (const std::bad_alloc&) {
        return 4;
    } catch (...) {
        return 5;
    }
}

PyObject* candidates(PyObject*, PyObject* args) {
    PyObject *first, *second, *support;
    Grid grid{};
    if (!PyArg_ParseTuple(args, "OOOiiid", &first, &second, &support, &grid.bins,
                          &grid.maximum_x, &grid.maximum_y, &grid.required_fraction))
        return nullptr;
    Buffer r, g, mask;
    if (!r.acquire(first) || !g.acquire(second) || !mask.acquire(support)) return nullptr;
    const Py_ssize_t h = r.view.shape[0], w = r.view.shape[1];
    // Bound all integer indices/products before narrowing; no untrusted dimensions in C++.
    if (h <= 0 || w <= 0 || h > INT_MAX / 2 || w > INT_MAX / 2 ||
        h > INT_MAX / w || g.view.shape[0] != h || g.view.shape[1] != w ||
        mask.view.shape[0] != h || mask.view.shape[1] != w ||
        r.view.len != h * w || g.view.len != h * w || mask.view.len != h * w ||
        grid.bins < 8 || grid.bins > 256 || grid.maximum_x < 0 || grid.maximum_y < 0 ||
        grid.maximum_x >= w || grid.maximum_y >= h ||
        (2LL * grid.maximum_x + 1) * (2LL * grid.maximum_y + 1) > INT_MAX ||
        !std::isfinite(grid.required_fraction) || grid.required_fraction <= 0 ||
        grid.required_fraction > 1) {
        PyErr_SetString(PyExc_ValueError, "Invalid NMI geometry, bins, range or support fraction");
        return nullptr;
    }
    grid.height = static_cast<int>(h); grid.width = static_cast<int>(w);
    std::vector<double> scores;
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = calculate(static_cast<const uint8_t*>(r.view.buf),
                       static_cast<const uint8_t*>(g.view.buf),
                       static_cast<const uint8_t*>(mask.view.buf), grid, scores);
    Py_END_ALLOW_THREADS
    switch (status) {
        case 0: break;
        case 1: PyErr_SetString(PyExc_ValueError, "Mutual-information overlap is insufficient"); return nullptr;
        case 2: PyErr_SetString(PyExc_ValueError, "Mutual-information input lacks entropy"); return nullptr;
        case 3: PyErr_SetString(PyExc_ValueError, "NMI quantised values exceed histogram bins"); return nullptr;
        case 4: return PyErr_NoMemory();
        default: PyErr_SetString(PyExc_RuntimeError, "Native NMI calculation failed"); return nullptr;
    }
    PyObject* result = PyList_New(0);
    if (!result) return nullptr;
    const int columns = 2 * grid.maximum_x + 1;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (!std::isfinite(scores[i])) continue;
        PyObject* row = Py_BuildValue("dii", scores[i], int(i % columns) - grid.maximum_x,
                                     int(i / columns) - grid.maximum_y);
        if (!row || PyList_Append(result, row)) {
            Py_XDECREF(row); Py_DECREF(result); return nullptr;
        }
        Py_DECREF(row);
    }
    return result;
}

PyMethodDef methods[] = {
    {"candidates", candidates, METH_VARARGS, "Exact exhaustive masked NMI candidate grid."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_rg_nmi", "Bounded exhaustive NMI accelerator.",
    -1, methods, nullptr, nullptr, nullptr, nullptr
};
} // namespace

PyMODINIT_FUNC PyInit__rg_nmi() { return PyModule_Create(&module); }
