from multiprocessing import Process, Pipe
import time

def src(input_send_conn, iter):
    for _ in range(iter):
        input_send_conn.send(1)
    input_send_conn.send(None)

def sink(output_recv_conn):
    while True:
        result = output_recv_conn.recv()
        if not result:
            break
    print(result)

def func(recv_conn, send_conn, iter):
    while True:
        data = recv_conn.recv()
        if data:
            for i in range(iter):
                data += 1
            send_conn.send(data)
        else:
            send_conn.send(data)
            break

if __name__ == "__main__":
    # 1 single process 
    input_send_conn, input_recv_conn = Pipe()
    output_send_conn, output_recv_conn = Pipe()
    result = []
    src_iter = 1000
    func_iter = 100000

    p1 = Process(target=func, args=(input_recv_conn, output_send_conn, func_iter))
    p_src = Process(target=src, args=(input_send_conn, src_iter))
    p_sink = Process(target=sink, args=(output_recv_conn,))
    
    p1.start()
    p_sink.start()
    time.sleep(10)
    stime = time.time()
    p_src.start()

    p1.join()
    p_src.join()
    p_sink.join()
    etime = time.time()

    print(f"Single process {etime - stime}")


    # 2 Dual Process : Pipelining
    input_send_conn, input_recv_conn = Pipe()
    mid_send_conn, mid_recv_conn = Pipe()
    output_send_conn, output_recv_conn = Pipe()
    result = []
    src_iter = 1000
    func_iter = 100000
    p1 = Process(target=func, args=(input_recv_conn, mid_recv_conn, func_iter//2))
    p2 = Process(target=func, args=(mid_send_conn, output_send_conn, func_iter//2))
    p_src = Process(target=src, args=(input_send_conn, src_iter))
    p_sink = Process(target=sink, args=(output_recv_conn,))

    p1.start()
    p2.start()
    p_sink.start()
    time.sleep(10)
    stime = time.time()
    p_src.start()

    p1.join()
    p2.join()
    p_src.join()
    p_sink.join()
    etime = time.time()
    
    print(f"dual process {etime - stime}")
