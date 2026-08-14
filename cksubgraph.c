#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ERR_NONE 0
#define ERR_OOM 1
#define ERR_DMAX 2
#define ERR_BAD_GRAPH 3
#include <time.h>
#include <stdio.h>





static unsigned default_seed(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    static unsigned counter = 0;
    return (unsigned)ts.tv_nsec ^ (unsigned)ts.tv_sec ^ counter++;
}

typedef struct {
    int n;
    int max_deg;
    const int *adj;
    const int *deg;

    int k;

    int *state;      /* shape [k] */
    int *pos;       // [n] inverse index - contains i in state or -1 (if node is not in subgraph)

    int *frontier;         /* shape [n] */
    unsigned int *frontier_stamp; /* shape [n] */
    unsigned int cur_frontier_stamp;

    int *queue;      /* shape [k] */
    int *comp_id;    /* shape [k] */
    unsigned char *seen_comp; /* shape [k] */

    int *move_u_idx; /* dynamic */
    int *move_v;     /* dynamic */

    size_t move_cap;
    size_t move_count;

    uint64_t rng_state;
} Sampler;

static uint64_t next_rand(Sampler *s) {
    uint64_t x = s->rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    s->rng_state = x;
    return x;
}


static double rng_uniform01(Sampler *s) {
    return (double)(next_rand(s) >> 11) * (1.0 / 9007199254740992.0); /* top 53 bits */
}

static int rng_below(Sampler *s, uint64_t bound) {
    return (int)(next_rand(s) % bound);   /* tiny modulo bias, irrelevant here */
}

static void sampler_free(Sampler *s) {
    free(s->state);
    free(s->pos);
    free(s->frontier);
    free(s->frontier_stamp);
    free(s->queue);
    free(s->comp_id);
    free(s->seen_comp);
    free(s->move_u_idx);
    free(s->move_v);
    memset(s, 0, sizeof(*s));
}


static int init_sampler(
    Sampler *s,
    int n,
    int max_deg,
    const int *adj,
    const int *deg,
    int k,
    const int *init_state,
    uint64_t seed
) {
    memset(s, 0, sizeof(*s));
    s->n = n;
    s->max_deg = max_deg;
    s->adj = adj;
    s->deg = deg;
    s->k = k;
    s->rng_state = seed ? seed : (uint64_t) default_seed();

    if (s->rng_state == 0) s->rng_state = 0x9E3779B97F4A7C15ULL;

    // s->rng_state = seed;
    s->cur_frontier_stamp = 1U;

    s->state = (int *)malloc((size_t)k * sizeof(int));
    s->pos = (int *)malloc((size_t)n * sizeof(int));
    s->frontier = (int *)malloc((size_t)n * sizeof(int));
    s->frontier_stamp = (unsigned int *)calloc((size_t)n, sizeof(unsigned int));
    s->queue = (int *)malloc((size_t)k * sizeof(int));
    s->comp_id = (int *)malloc((size_t)k * sizeof(int));
    s->seen_comp = (unsigned char *)malloc((size_t)k * sizeof(unsigned char));


    size_t move_cap = (size_t)k * (size_t)(n - k);

    s->move_cap = move_cap;


    if (move_cap > 0) {
        s->move_u_idx = (int *)malloc(move_cap * sizeof(int));
        s->move_v = (int *)malloc(move_cap * sizeof(int));

        if (s->move_u_idx == NULL || s->move_v == NULL) {
            sampler_free(s);
            return 0;
        }
    }

    if (s->state == NULL || s->pos == NULL || s->frontier == NULL ||
        s->frontier_stamp == NULL || s->queue == NULL ||
        s->comp_id == NULL || s->seen_comp == NULL) {
        sampler_free(s);
        return 0;
    }

    for (int v = 0; v < n; ++v) {
        s->pos[v] = -1;
    }

    for (int i = 0; i < k; ++i) {
        int u = init_state[i];
        if (u < 0 || u >= n) {
            sampler_free(s);
            return 0;
        }
        if (s->pos[u] != -1) {
            sampler_free(s);
            return 0;
        }
        s->state[i] = u;
        s->pos[u] = i;
    }

    return 1;
}

static int build_frontier(Sampler *s) {
    if (++s->cur_frontier_stamp == 0U) {
        memset(s->frontier_stamp, 0, (size_t)s->n * sizeof(unsigned int));
        s->cur_frontier_stamp = 1U;
    } // this is *extremely* defensive, but whatever;

    int frontier_count = 0;
    for (int i = 0; i < s->k; ++i) {
        int u = s->state[i];
        const size_t row = (size_t)u * (size_t)s->max_deg;
        for (int t = 0; t < s->deg[u]; ++t) {
            int w = s->adj[row + (size_t)t];
            if (s->pos[w] == -1 && s->frontier_stamp[w] != s->cur_frontier_stamp) {
                s->frontier_stamp[w] = s->cur_frontier_stamp;
                s->frontier[frontier_count++] = w;
            }
        }
    }
    return frontier_count;
}


static int label_components_after_removal(Sampler *s, int skip_i) {
    for (int i = 0; i < s->k; ++i) {
        s->comp_id[i] = -1;
    }

    int num_comp = 0;

    // suppose that we removed skip_i?
    for (int seed_i = 0; seed_i < s->k; ++seed_i) {
        if (seed_i == skip_i || s->comp_id[seed_i] != -1) {
            continue;
        }

        int qh = 0;
        int qt = 0;
        s->comp_id[seed_i] = num_comp;
        s->queue[qt++] = seed_i;

        while (qh < qt) {
            int cur_i = s->queue[qh++];
            int u = s->state[cur_i];
            const size_t row = (size_t)u * (size_t)s->max_deg;

            for (int t = 0; t < s->deg[u]; ++t) {
                int w = s->adj[row + (size_t)t];
                int j = s->pos[w];
                if (j >= 0 && j != skip_i && s->comp_id[j] == -1) {
                    s->comp_id[j] = num_comp;
                    s->queue[qt++] = j;
                }
            }
        }

        num_comp += 1;
    }

    return num_comp;
}

static int recompute_moves(Sampler *s) {
    const int frontier_count = build_frontier(s);

    s->move_count = 0;

    for (int skip_i = 0; skip_i < s->k; ++skip_i) {
        int num_comp = label_components_after_removal(s, skip_i);
        if (num_comp <= 0) {
            continue;
        }

        for (int f = 0; f < frontier_count; ++f) {
            int v = s->frontier[f];
            memset(s->seen_comp, 0, (size_t)num_comp * sizeof(unsigned char));
            int seen_count = 0;

            const size_t row = (size_t)v * (size_t)s->max_deg;
            for (int t = 0; t < s->deg[v]; ++t) {
                int w = s->adj[row + (size_t)t];
                int j = s->pos[w];
                if (j >= 0 && j != skip_i) {
                    int cid = s->comp_id[j];
                    if (cid >= 0 && !s->seen_comp[cid]) {
                        s->seen_comp[cid] = 1;
                        seen_count += 1;
                        if (seen_count == num_comp) {
                            break;
                        }
                    }
                }
            }

            if (seen_count == num_comp) {
                size_t idx = s->move_count++;
                s->move_u_idx[idx] = skip_i;
                s->move_v[idx] = v;
            }
        }
    }

    return ERR_NONE;
}

static int state_is_connected(Sampler *s) {
    if (s->k <= 1) {
        return 1;
    }
    return label_components_after_removal(s, -1) == 1;
}

static int cmp_int_desc(const void *a, const void *b) {
    const int ia = *(const int *)a;
    const int ib = *(const int *)b;
    return (ib > ia) - (ib < ia);
}

static uint64_t compute_dmax_tighter(const int *deg, int n, int k) {
    int *tmp = (int *)malloc((size_t)n * sizeof(int));
    if (tmp == NULL) {
        return 0;
    }
    memcpy(tmp, deg, (size_t)n * sizeof(int));
    qsort(tmp, (size_t)n, sizeof(int), cmp_int_desc);

    uint64_t topk_sum = 0;
    for (int i = 0; i < k; ++i) {
        topk_sum += (uint64_t)tmp[i];
    }
    free(tmp);

    uint64_t correction = (uint64_t)2 * (uint64_t)(k - 1);
    uint64_t frontier_bound = (topk_sum > correction) ? (topk_sum - correction) : 0ULL;
    return (uint64_t)k * frontier_bound;
}

static PyObject *py_sample_lazy(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *adj_obj = NULL;
    PyObject *deg_obj = NULL;
    PyObject *init_obj = NULL;
    PyObject *dmax_obj = Py_None;
    PyObject *seed_obj = Py_None;
    Py_ssize_t iterations = 0;

    static char *kwlist[] = {
        "adj", "deg", "init_state", "iterations", "d_max", "seed", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOn|OO",
            kwlist,
            &adj_obj,
            &deg_obj,
            &init_obj,
            &iterations,
            &dmax_obj,
            &seed_obj)) {
        return NULL;
    }

    if (iterations < 0) {
        PyErr_SetString(PyExc_ValueError, "iterations must be non-negative");
        return NULL;
    }

    PyArrayObject *adj_arr = (PyArrayObject *)PyArray_FROM_OTF(
        adj_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY
    );
    PyArrayObject *deg_arr = (PyArrayObject *)PyArray_FROM_OTF(
        deg_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY
    );
    PyArrayObject *init_arr = (PyArrayObject *)PyArray_FROM_OTF(
        init_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY
    );

    if (adj_arr == NULL || deg_arr == NULL || init_arr == NULL) {
        Py_XDECREF(adj_arr);
        Py_XDECREF(deg_arr);
        Py_XDECREF(init_arr);
        return NULL;
    }

    if (PyArray_NDIM(adj_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "adj must be a 2D int32 array of shape [n, max_deg]");
        goto fail;
    }
    if (PyArray_NDIM(deg_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "deg must be a 1D int32 array of shape [n]");
        goto fail;
    }
    if (PyArray_NDIM(init_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "init_state must be a 1D int32 array of shape [k]");
        goto fail;
    }

    const int n = (int)PyArray_DIM(adj_arr, 0);
    const int max_deg = (int)PyArray_DIM(adj_arr, 1);
    if ((int)PyArray_DIM(deg_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "deg must have length n = adj.shape[0]");
        goto fail;
    }

    const int k = (int)PyArray_DIM(init_arr, 0);
    if (k < 2) {
        PyErr_SetString(PyExc_ValueError, "k must be at least 2 in this implementation");
        goto fail;
    }
    if (k > n) {
        PyErr_SetString(PyExc_ValueError, "init_state length cannot exceed number of vertices");
        goto fail;
    }

    const int *adj = (const int *)PyArray_DATA(adj_arr);
    const int *deg = (const int *)PyArray_DATA(deg_arr);
    const int *init_state = (const int *)PyArray_DATA(init_arr);

    uint64_t d_max = 0ULL;
    if (dmax_obj == Py_None) {
        d_max = compute_dmax_tighter(deg, n, k);
        if (d_max == 0ULL) {
            PyErr_SetString(PyExc_ValueError, "computed d_max is zero; check the graph or pass d_max explicitly");
            goto fail;
        }
    } else {
        d_max = PyLong_AsUnsignedLongLong(dmax_obj);
        if (PyErr_Occurred()) {
            goto fail;
        }
        if (d_max == 0ULL) {
            PyErr_SetString(PyExc_ValueError, "d_max must be positive");
            goto fail;
        }
    }

    uint64_t seed = 0ULL;
    if (seed_obj != Py_None) {
        seed = PyLong_AsUnsignedLongLong(seed_obj);
        if (PyErr_Occurred()) {
            goto fail;
        }
    } else {
        seed = (uint64_t)default_seed();
    }

    Sampler s;
    if (!init_sampler(&s, n, max_deg, adj, deg, k, init_state, seed)) {
        PyErr_SetString(PyExc_ValueError, "failed to initialize sampler; check init_state uniqueness and bounds");
        goto fail;
    }

    if (!state_is_connected(&s)) {
        sampler_free(&s);
        PyErr_SetString(PyExc_ValueError, "init_state must induce a connected subgraph");
        goto fail;
    }

    int status = recompute_moves(&s);
    if (status == ERR_OOM) {
        sampler_free(&s);
        PyErr_NoMemory();
        goto fail;
    }
    if ((uint64_t)s.move_count > d_max) {
        sampler_free(&s);
        PyErr_SetString(PyExc_ValueError, "d_max is too small for the initial state");
        goto fail;

    }

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t it = 0; it < iterations; ++it) {
        if (s.move_count > 0 &&  rng_uniform01(&s) < ((double)s.move_count / (double)d_max)) {
            size_t which = (size_t)rng_below(&s, (uint64_t)s.move_count);
            int i = s.move_u_idx[which];
            int old_u = s.state[i];
            int new_v = s.move_v[which];

            s.pos[old_u] = -1;
            s.state[i] = new_v;
            s.pos[new_v] = i;

            status = recompute_moves(&s);
            if (status != ERR_NONE) {
                break;
            }
            if ((uint64_t)s.move_count > d_max) {
                status = ERR_DMAX;
                break;
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (status == ERR_OOM) {
        sampler_free(&s);
        PyErr_NoMemory();
        goto fail;
    }
    if (status == ERR_DMAX) {
        sampler_free(&s);
        PyErr_SetString(PyExc_ValueError, "d_max is too small for a visited state");
        goto fail;
    }
    if (status == ERR_BAD_GRAPH) {
        sampler_free(&s);
        PyErr_SetString(PyExc_RuntimeError, "internal graph error");
        goto fail;
    }

    npy_intp dims[1] = {(npy_intp)k};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
    if (out == NULL) {
        sampler_free(&s);
        goto fail;
    }
    memcpy(PyArray_DATA(out), s.state, (size_t)k * sizeof(int));
    sampler_free(&s);

    Py_DECREF(adj_arr);
    Py_DECREF(deg_arr);
    Py_DECREF(init_arr);
    return (PyObject *)out;

fail:
    Py_XDECREF(adj_arr);
    Py_XDECREF(deg_arr);
    Py_XDECREF(init_arr);
    return NULL;
}


static PyObject *py_sample_lazy_logged(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *adj_obj = NULL, *deg_obj = NULL, *init_obj = NULL;
    PyObject *dmax_obj = Py_None, *seed_obj = Py_None;
    Py_ssize_t iterations = 0, thin = 1;
    int check_invariants = 0;

    PyArrayObject *adj_arr = NULL, *deg_arr = NULL, *init_arr = NULL;
    PyArrayObject *traj_arr = NULL, *mcount_arr = NULL, *final_arr = NULL;

    static char *kwlist[] = {
        "adj", "deg", "init_state", "iterations",
        "d_max", "seed", "thin", "check_invariants", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOn|OOnp", kwlist,
            &adj_obj, &deg_obj, &init_obj, &iterations,
            &dmax_obj, &seed_obj, &thin, &check_invariants)) {
        return NULL;
    }
    if (iterations < 0) { PyErr_SetString(PyExc_ValueError, "iterations must be non-negative"); return NULL; }
    if (thin < 1)       { PyErr_SetString(PyExc_ValueError, "thin must be >= 1"); return NULL; }

    adj_arr  = (PyArrayObject *)PyArray_FROM_OTF(adj_obj,  NPY_INT32, NPY_ARRAY_IN_ARRAY);
    deg_arr  = (PyArrayObject *)PyArray_FROM_OTF(deg_obj,  NPY_INT32, NPY_ARRAY_IN_ARRAY);
    init_arr = (PyArrayObject *)PyArray_FROM_OTF(init_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (adj_arr == NULL || deg_arr == NULL || init_arr == NULL) goto fail;

    if (PyArray_NDIM(adj_arr) != 2)  { PyErr_SetString(PyExc_ValueError, "adj must be 2D [n, max_deg]"); goto fail; }
    if (PyArray_NDIM(deg_arr) != 1)  { PyErr_SetString(PyExc_ValueError, "deg must be 1D [n]"); goto fail; }
    if (PyArray_NDIM(init_arr) != 1) { PyErr_SetString(PyExc_ValueError, "init_state must be 1D [k]"); goto fail; }

    const int n = (int)PyArray_DIM(adj_arr, 0);
    const int max_deg = (int)PyArray_DIM(adj_arr, 1);
    if ((int)PyArray_DIM(deg_arr, 0) != n) { PyErr_SetString(PyExc_ValueError, "deg must have length n"); goto fail; }

    const int k = (int)PyArray_DIM(init_arr, 0);
    if (k < 2) { PyErr_SetString(PyExc_ValueError, "k must be at least 2"); goto fail; }
    if (k > n) { PyErr_SetString(PyExc_ValueError, "k cannot exceed n"); goto fail; }

    const int *adj = (const int *)PyArray_DATA(adj_arr);
    const int *deg = (const int *)PyArray_DATA(deg_arr);
    const int *init_state = (const int *)PyArray_DATA(init_arr);

    uint64_t d_max = 0ULL;
    if (dmax_obj == Py_None) {
        d_max = compute_dmax_tighter(deg, n, k);
        if (d_max == 0ULL) { PyErr_SetString(PyExc_ValueError, "computed d_max is zero"); goto fail; }
    } else {
        d_max = PyLong_AsUnsignedLongLong(dmax_obj);
        if (PyErr_Occurred()) goto fail;
        if (d_max == 0ULL) { PyErr_SetString(PyExc_ValueError, "d_max must be positive"); goto fail; }
    }

    uint64_t seed = (seed_obj != Py_None) ? PyLong_AsUnsignedLongLong(seed_obj)
                                          : (uint64_t)default_seed();
    if (PyErr_Occurred()) goto fail;

    Sampler s;
    if (!init_sampler(&s, n, max_deg, adj, deg, k, init_state, seed)) {
        PyErr_SetString(PyExc_ValueError, "failed to initialize sampler"); goto fail;
    }
    if (!state_is_connected(&s)) {
        sampler_free(&s);
        PyErr_SetString(PyExc_ValueError, "init_state must induce a connected subgraph"); goto fail;
    }

    recompute_moves(&s);
    int status = ERR_NONE;
    if ((uint64_t)s.move_count > d_max) {
        sampler_free(&s);
        PyErr_SetString(PyExc_ValueError, "d_max is too small for the initial state"); goto fail;
    }

    /* allocate output buffers BEFORE releasing the GIL; we own them, so writing
       their raw data without the GIL is safe */
    size_t num_records = (size_t)(iterations / thin);
    npy_intp tdims[2] = {(npy_intp)num_records, (npy_intp)k};
    npy_intp mdims[1] = {(npy_intp)num_records};
    traj_arr   = (PyArrayObject *)PyArray_SimpleNew(2, tdims, NPY_INT32);
    mcount_arr = (PyArrayObject *)PyArray_SimpleNew(1, mdims, NPY_INT64);
    if (traj_arr == NULL || mcount_arr == NULL) { sampler_free(&s); goto fail; }

    int32_t *traj   = (int32_t *)PyArray_DATA(traj_arr);
    int64_t *mcount = (int64_t *)PyArray_DATA(mcount_arr);

    uint64_t accepted = 0;
    uint64_t max_mc = (uint64_t)s.move_count;
    double sum_acc = 0.0;
    size_t rec = 0;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t it = 0; it < iterations; ++it) {
        double p = (double)s.move_count / (double)d_max;   /* accept prob this step */
        sum_acc += p;

        if (s.move_count > 0 && rng_uniform01(&s) < p) {
            size_t which = (size_t)rng_below(&s, (uint64_t)s.move_count);
            int i = s.move_u_idx[which];
            int old_u = s.state[i];
            int new_v = s.move_v[which];
            s.pos[old_u] = -1;
            s.state[i] = new_v;
            s.pos[new_v] = i;
            recompute_moves(&s);
            accepted++;
            if ((uint64_t)s.move_count > max_mc) max_mc = (uint64_t)s.move_count;
            if ((uint64_t)s.move_count > d_max) { status = ERR_DMAX; break; }
            if (check_invariants && !state_is_connected(&s)) { status = ERR_BAD_GRAPH; break; }
        }

        if (((it + 1) % thin) == 0 && rec < num_records) {
            memcpy(traj + rec * (size_t)k, s.state, (size_t)k * sizeof(int32_t));
            mcount[rec] = (int64_t)s.move_count;
            rec++;
        }
    }
    Py_END_ALLOW_THREADS

    if (status == ERR_DMAX)      { sampler_free(&s); PyErr_SetString(PyExc_ValueError, "d_max too small for a visited state"); goto fail; }
    if (status == ERR_BAD_GRAPH) { sampler_free(&s); PyErr_SetString(PyExc_RuntimeError, "invariant violated: disconnected state produced"); goto fail; }

    npy_intp fdims[1] = {(npy_intp)k};
    final_arr = (PyArrayObject *)PyArray_SimpleNew(1, fdims, NPY_INT32);
    if (final_arr == NULL) { sampler_free(&s); goto fail; }
    memcpy(PyArray_DATA(final_arr), s.state, (size_t)k * sizeof(int32_t));
    sampler_free(&s);

    PyObject *result = PyDict_New();
    if (result == NULL) goto fail;

    PyObject *tmp;
    PyDict_SetItemString(result, "final_state",      (PyObject *)final_arr);
    PyDict_SetItemString(result, "trajectory",       (PyObject *)traj_arr);
    PyDict_SetItemString(result, "move_count_trace", (PyObject *)mcount_arr);
    tmp = PyLong_FromSsize_t(iterations);            PyDict_SetItemString(result, "iterations", tmp);            Py_DECREF(tmp);
    tmp = PyLong_FromUnsignedLongLong(accepted);     PyDict_SetItemString(result, "accepted_moves", tmp);        Py_DECREF(tmp);
    tmp = PyFloat_FromDouble(sum_acc);               PyDict_SetItemString(result, "sum_acceptance_prob", tmp);   Py_DECREF(tmp);
    tmp = PyFloat_FromDouble(iterations > 0 ? sum_acc / (double)iterations : 0.0);
                                                     PyDict_SetItemString(result, "mean_acceptance_prob", tmp);  Py_DECREF(tmp);
    tmp = PyLong_FromUnsignedLongLong(max_mc);       PyDict_SetItemString(result, "max_move_count", tmp);        Py_DECREF(tmp);
    tmp = PyLong_FromUnsignedLongLong(d_max);        PyDict_SetItemString(result, "d_max", tmp);                 Py_DECREF(tmp);
    tmp = PyLong_FromSsize_t(thin);                  PyDict_SetItemString(result, "thin", tmp);                  Py_DECREF(tmp);

    Py_DECREF(final_arr);
    Py_DECREF(traj_arr);
    Py_DECREF(mcount_arr);
    Py_DECREF(adj_arr);
    Py_DECREF(deg_arr);
    Py_DECREF(init_arr);
    return result;

fail:
    Py_XDECREF(final_arr);
    Py_XDECREF(traj_arr);
    Py_XDECREF(mcount_arr);
    Py_XDECREF(adj_arr);
    Py_XDECREF(deg_arr);
    Py_XDECREF(init_arr);
    return NULL;
}

static PyMethodDef module_methods[] = {
    {
        "sample_lazy",
        (PyCFunction)py_sample_lazy,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR(
            "sample_lazy(adj, deg, init_state, iterations, d_max=None, seed=None) -> ndarray\n"
            "\n"
            "Lazy MCMC on connected induced k-subgraphs using a padded adjacency array.\n"
            "adj: int32 array of shape [n, max_deg]\n"
            "deg: int32 array of shape [n]\n"
            "init_state: int32 array of shape [k], must be connected and unique\n"
            "iterations: number of lazy-chain steps\n"
            "d_max: optional positive global upper bound on the move count\n"
            "seed: optional uint64 seed\n"
        )
    },
{
    "sample_lazy_logged",
    (PyCFunction)py_sample_lazy_logged,
    METH_VARARGS | METH_KEYWORDS,
    PyDoc_STR(
        "sample_lazy_logged(adj, deg, init_state, iterations, d_max=None, "
        "seed=None, thin=1, check_invariants=False) -> dict\n"
        "Like sample_lazy, but returns a dict with the time-sampled trajectory "
        "(every `thin` steps, self-loops included), move_count_trace, and counters: "
        "iterations, accepted_moves, sum/mean_acceptance_prob, max_move_count, d_max."
    )
},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cksubgraph",
    "Connected k-subgraph lazy MCMC sampler.",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_cksubgraph(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
