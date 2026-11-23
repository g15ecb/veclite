from kernel import OK, E_FILTER, bptr, imut

def check_device(device: Int32, min_batch: Int64, nq: Int64, filter_on: Int, mut filtered: Bool) -> Int32:
    if device < DEV_AUTO or device > DEV_GPU or min_batch < 1:
        return E_ARG
    filtered = False
    if filter_on != 0:
        for q in range(Int(nq)):
            var f = bptr(filter_on)[unsafe_offset=q]
            if f > 1:
                return E_FILTER
            filtered = filtered or f == 1
    return OK


@export("veclite_gpu_available")
def veclite_gpu_available() abi("C") -> Int32:
    return 0


@export("veclite_close")
def veclite_close(h: Int) abi("C") -> Int32:
    return close_db(h)


@export("veclite_reader_search_dev")
def veclite_reader_search_dev(
    h: Int, r: Int64, queries: Int, nq: Int64, filter_on: Int, filter_tags: Int, k: Int64, metric: Int32, nchunks: Int64,
    device: Int32, min_batch: Int64, out_ids: Int, out_scores: Int, out_counts: Int, out_used: Int,
) abi("C") -> Int32:
    if out_used != 0:
        Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=out_used)[unsafe_offset=0] = 0
    var st = check_search(h, r, queries, nq, k, out_ids, out_scores, out_counts)
    var filtered = False
    if st == OK:
        st = check_device(device, min_batch, nq, filter_on, filtered)
    if st == OK and device == DEV_GPU:
        st = E_NO_GPU
    if st != OK:
        return st
    return search_cpu(h, r, queries, nq, filter_on, filter_tags, k, metric, nchunks, out_ids, out_scores, out_counts)
