from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size == 2, "Run with exactly 2 ranks"

N = 1_000_000  # bytes
buf = bytearray(N)

# warmup
for _ in range(10):
    if rank == 0:
        comm.Send([buf, MPI.BYTE], dest=1)
        comm.Recv([buf, MPI.BYTE], source=1)
    else:
        comm.Recv([buf, MPI.BYTE], source=0)
        comm.Send([buf, MPI.BYTE], dest=0)

# timed
iters = 100
t0 = time.perf_counter()
for _ in range(iters):
    if rank == 0:
        comm.Send([buf, MPI.BYTE], dest=1)
        comm.Recv([buf, MPI.BYTE], source=1)
    else:
        comm.Recv([buf, MPI.BYTE], source=0)
        comm.Send([buf, MPI.BYTE], dest=0)
t1 = time.perf_counter()

if rank == 0:
    rtt = (t1 - t0) / iters
    one_way = rtt / 2
    mbps = (N / one_way) / (1024 * 1024)
    print(f"ping-pong RTT: {rtt*1e3:.3f} ms, one-way ~{one_way*1e3:.3f} ms, effective ~{mbps:.1f} MiB/s")
