---
title: "The Google File System"
date: 2020-06-02T09:53:35+01:00
---

This is my first look at one of the classic Google papers, [The Google File System](https://static.googleusercontent.com/media/research.google.com/en//archive/gfs-sosp2003.pdf). The lead author is Sanjay Ghemawat, one of the subjects of a [delightful New Yorker article](https://www.newyorker.com/magazine/2018/12/10/the-friendship-that-made-google-huge).

The paper is about the first iteration of the Google File System, all the way back in 2003. I know that the system has been rewritten since then (Collosus). All of the numbers these days are going to be orders of magnitude higher as well. So I think the best way for me to read this paper is to pretend that we're working at Google (pre-IPO at this stage) and we have some data-heavy MapReduce jobs (e.g. indexing a quickly growing internet) and we want them to be provided a scalable, performant, and fault-tolerant filesystem. In terms of numbers, we are talking about filesystem clusters which consist of thousands of machines, storing hundreds of terabytes in aggregate, access by hundreds of clients (which I imagine was exceptional in 2003).

The requirements and design assumptions are pretty interesting:
* Fault-tolerance: This one is obvious. Google were famously one of the first companies to run on commodity hardware, so certainly they need to design a system which can detect and respond from hardware (or software, or operator) failures. 
* Large files: The system is really designed for processing files which are several gigabytes large, for example files containing a bunch of web documents being processed during an indexing MapReduce job. The system should be able to handle small files, but needn't optimise for them.
* Reads are typically either large streaming reads (hundreds of KBs in sequence) or small random reads (a few KB at some arbitrary offset).
* Writes are typically large sequential appends. Small modifications at arbitrary offset must be supported but needn't be optimised. High sustained bandwith writes are more important than low-latency writes.
* The system must support hundreds of clients trying to append to the same file simultaneously.

## What Does It Look Like?

The filesystem API is pretty standard, but not actually POSIX. The filesystem is laid out with directories and pathnames and it supports `create`/`delete`/`open`/`close`/`read`/`write` as normal. It also supports a couple of non-standard operations, which we'll cover in more detail later.

* `snapshot`: Low-cost copy of file or directory tree
* `record`: Allows multiple clients to append to file

The high-level flow is: 

> Files are split up in to fixed 64MB _chunks_. The chunks are stored on _chunkservers_. The cluster consists of a single _master_ and a number of _chunkservers_. Clients connect to the master and request a file. Since they know what byte range they want to read from that file, they can work out client side which chunk(s) they want. The master responds with information about on which chunkserver, and where on that chunkserver, the chunks that the client needs to read.

### Chunks In More Detail

Each chunk has a 64-bit id called a _chunk handle_. When these chunks are stored on chunkservers, they're just regular Linux files operated on in the usual way. Each chunk is replicated (thrice, by default) on a number of chunkservers.

Why are the chunks so large?

* Large chunks means the clients need to interact with the master less
* Reduce network overhead involved in jumping around in a file - since the chunk is so large, they're probably staying within the same chunk so can keep using their existing connection
* Less metadata needed on the master

They discuss contention at this point. As I see it, there are two points here. There's avoidable contention, which would be if many clients wanted to read at different offsets from the same file. With a smaller chunk size, you could distribute this work across more nodes. There's also unavoidable contention, like the example where a binary required by many workers was written to the filesystem. (To circumvent this, they increased the replication factor.)

### The Master In More Detail

The master is responsible for pretty much everything except actually reading the files. This includes storing the filesystem metadata, the mapping from file to chunks, etc. It also performs cluster maintenance tasks, like health-checking chunkservers and migrating chunks between chunkservers. There isn't really any IO happening on the master, but going to be the bottleneck for basically everything else. Steps are taken to minimise this, for example clients to cache their responses from the master, so they can probably read nearby to where they originally asked without having to interact with the master again.

The most useful data (file/chunk namespaces, file-to-chunk mapping, location of chunks) are kept in memory on the master. Most of this is also replicated to disk on masters in its _operation log_, whcih is also replicated throughout the cluster. The exception is the location of the chunks: the master gets thee simply by asking chunkservers, and if a new master were elected, it would recover this data in the same way. Since the operation log is critical to GFS, a write has to be persisted to all replicas of the log before it is successful. To improve throughput there is batching of log entries, and to improve startup time there is log checkpointing.

Since the master is responsible for file namesapce mutations (like file creation), it's not surprising that GFS guarantees that these operations are atomic.

### Consistency In The Files

The master has a pretty consistent hold on the metadata, but what actually ends up in the files is much spicier. GFS distinguishes `writes` (just a standard write, at some application-defined offset) and `record appends` (causes the data to be appended atomically at least once at an offset of GFS choosing). The latter is the sort of thing you might use if there are multiple concurrent writers but you just want to ensure your data ends up _somewhere_ near the end of the file.

Standard `writes` work fine for serial access (the paper calls this _defined_ - all reads will return the same data, and moreover that data reflects the result of that write). For concurrent `writes`, GFS guarantees _consistency_ in the sense that all reads will return the same data, but that data need not be what any individual writer wrote ("typically a mingled fragment from multiple mutations").

For `record appends`, both serially and concurrently, everything ends up "_defined_ interspersed with _inconistent_". Basically this is as above, all the writes end up in the file as a contiguous chunks of data, but there's going to be some mess around it. This is the typical use-case, and clients just need to be resilient to this potential mess. There's library code available to applications in Google to work around these issues.

## Leases and Mutation Order

Given that the data is replicated and the master is not involved, how can we actually give _any_ guarantees over consistency of writes? For this, the master grants a chunk lease to one of the replicas (the _primary_). The primary is responsible for picking a serial order for applying mutations to the chunk.

From the perspective of a client who wants to write some data, the flow is as follows. First they ask the master about the location of the chunks, this includes which replica is the primary. The client then pushes its data to all its replicas, who will store it in a buffer. Once acked by each replica, the client sends a write request to the primary. The primary might be dealing with multiple write requests here, but it decides on an order for them, applies the changes to its own copy of the chunk, then pushes the ordering to all other replicas. Once these are done and acked back to the primary, the primary informs the client of the happy outcome.

The "mingled fragments from multiple mutations" happens for example when the write payloads from the clients are large, as the client library will break these up in to multiple smaller write payloads. With many writers, the payloads can be intermingled. (This feels avoidable, but I guess there's an undiscussed tradeoff in enforcing per-client continuity when the primary orders the payloads.)
