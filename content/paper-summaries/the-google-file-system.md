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

## System Details

### Leases and Mutation Order

Given that the data is replicated and the master is not involved, how can we actually give _any_ guarantees over consistency of writes? For this, the master grants a chunk lease to one of the replicas (the _primary_). The primary is responsible for picking a serial order for applying mutations to the chunk.

From the perspective of a client who wants to write some data, the flow is as follows. First they ask the master about the location of the chunks, this includes which replica is the primary. The client then pushes its data to all its replicas, who will store it in a buffer. Once acked by each replica, the client sends a write request to the primary. The primary might be dealing with multiple write requests here, but it decides on an order for them, applies the changes to its own copy of the chunk, then pushes the ordering to all other replicas. Once these are done and acked back to the primary, the primary informs the client of the happy outcome.

The "mingled fragments from multiple mutations" happens for example when the write payloads from the clients are large, as the client library will break these up in to multiple smaller write payloads. With many writers, the payloads can be intermingled. (This feels avoidable, but I guess there's an undiscussed tradeoff in enforcing per-client continuity when the primary orders the payloads.)

### Data Flow

This part is pretty simple. Data is pushed linearly as possible: If a chunkserver has some data that it should forward, it forwards it to its nearest neighbour who also needs that data, and so on. There is no fanning out, which seems sensible in a situation where you're expecting large payloads. They mention typically having 100Mbps network links in the data centre, which shows this paper's age.

### Record Appends

We briefly discussed this operation, which semantically guarantees that your data will end up appended to the file as a contiguous chunk. This is a typically use case, for example with multiple workers using the file as a producer-consumer queue. It's similar to an ordinary write: the client pushes its data to all the replicas, then asks the primary to allow the write. Since it's a record append, though, the primary first checks if it would overflow the current chunk. If it would, it pads the current chunk and asks the client to try again. (Record appends are limited to 25% of the chunk size to mitigate the obvious problem with this.)

The operation is reported as successful iff the data is written as an atomic unit at the same offset in all replicas of some chunk. In reality, the record append could fail on some replica, in which case the client would have to retry.

### Snapshots

This is just a copy-on-write approach to copying a file. When the master receives the snapshot request for a file, it invalidates all the lease for all chunks in that file. In future, when it receives a modification request, it will ask all the chunkservers to replicate the chunks locally. (There are a few failure modes here that they don't discuss, but I don't think there's anything complicated going on.)

## More Master Responsibilities, High Avalability, and Fault Tolerance

There's a lot more details about the master, replication, and fault tolerance over the next few sections. I'll just pick a few highlights:

* Data should be replicated across racks. The pros: Can survive rack failure, can utilize bandwith from multiple racks simultaneously. The accepted con is that the data replication across racks is slower than replication within racks.
* The master makes the decision of where the chunks are replicated, taking in to account things like sharing load, spreading across racks, etc. It does this when chunks have to be created for whatever reason, but it also does this when machines die or things need to be rebalanced.
* Deletions are soft initially (rename the file to mark it as deleted) then hard after a few days. The master is solely responsible for whatever chunks should actually exist. It talks to the chunkservers and over time they will work out if that chunkserver has any chunks that are no longer on the master; the chunkserver can freely delete such chunks.
* For each chunk, the master maintains a chunk version number. Whenever the master grants a new lease on a chunk, it bumps the version number and informs all the replicas. If the master hears a chunkserver talking about a stale chunk version, it considers it deleted. If the master hears a chunkserver talking about a future chunk version, it trusts the chunkserver and updates its own record. (I'm a bit uncomfortable with this one, but I think this can only happen during master failover, so perhaps the previous master has ensured that there are enough replicas of the data around to avoid data loss.)
* The master is replicated for reliability: A mutation to the master's log is only successful when it has been written to disk on all master-replicas. Restarting the master is just replaying the log
* There are also "shadow-masters" that are read-only. They might slightly lag behind the master, but that's only in the metadata, which isn't updating very quickly. Having shadow masters spreads the read load.
* Chunkservers maintain checksums of their own data, stored in memory, so they can know if their disk has corrupted the data. A mismatch invalidates the chunk replica on that chunkserver, and means the master must reassign that replica. The checksumming method is optimized for appends.

## Wrapping Up

This is a long paper, covering a huge amount of topics. I skipped over a few. The "Experiences" section towards the end is worth a shout-out though.

They talk about the problem of working with disks that claim to support various IDE protocol versions but in fact quitely mishandle some edge cases, leaving the kernel and the driver in an inconsistent state. This sounds like a bit of a nigthmare, and is why the introduced checksums. They also mention a few pure kernel issues, but are also grateful that they can read the Linux source code and upstream improvements.
