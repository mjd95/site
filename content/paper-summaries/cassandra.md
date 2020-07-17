---
title: "Cassandra - A Decentralized Structured Storage System"
date: 2020-06-23T09:53:35+01:00
---

Lockdown is easing a bit in the UK and I'm starting to have something resembling a social life again, so I'm finding myself with a bit less time for reading papers. I've enjoyed regularly reading papers again though, so I'm going to try to keep this up at least once per week. I'll still try to keep up a mixture of computer science and machine learning.

Today I'm going to continue with my recent NoSQL streak, and look at the [Cassandra paper](https://www.cs.cornell.edu/Projects/ladis2009/papers/Lakshman-ladis2009.PDF). This is a pretty old paper again (2009). Cassandra isn't too fashionable these days, but it's undeniably been a pretty big player in the last decade and deserves some attention.

Cassandra was initially designed to solve the "Inbox Search" problem. Each user has an inbox of messages that has been sent to them, and the user would like to be able to search these. This would require the system to handle billions of writes per day, and the system should be able to scale up with new users. (In terms of user numbers, they had 100 million in 2008 at deployment time, and 250 million in 2009 when the paper was published. Of course this continued to grow exponentially for a long time.) Reads should be low-latency, which means the system is designed with cross data centre replication in mind. Over time, Cassandra became the storage backend for various other services.

## Related Work

They mention GFS, which I wrote about recently. (In fact, they use exactly the same words to describe GFS as the Dynamo paper did, which is a bit embarrassing). By the time this paper was written, the single-master aspect of GFS had been augmented with Chubby so it's fault-tolerant, but it's pretty much still the same design. In particular, the master is still the scalibility bottleneck. There's also BigTable built on top of this, which is only very briefly mentioned.

They go in to more detail on Dynamo (the original one, not DynamoDB) from Amazon. The main negative thing they say about Dynamo is that each write requires a read, which obviously impacts write performance. (This is due to the Dynamo vector clock system - clients need to know which version of the data they're updating.)

There's also a bunch of other systems mentioned, which are less famous (at least in industry today).

## Data Model

A table in Cassandra is a distributed, multi-dimensional map indexed by a key. The key is a string with no size contraints, but typically 16-36 bytes long. All operations on the values for a given key are atomic.

The values themselves are structured objects. Columns (the attributes appearing in the value) are grouped together in to sets called "column families", just like in Bigtable. There are both "simple" and "super" column families - "super" column families are column families within a column family. We'll see why the super column families are useful later on. Column families can be sorted by time or alphabetically.

At the time of wrigin, Cassandra was typically deployed as a cluster managing one table, owned by one application. 

## API

Cassandra has the following API methods:

* `inert(table, key, rowMutation)`
* `get(table, key, columnName)`
* `delete(table, key, columnName)`

In the last two examples, `columnName` could be a column within a column family, a column family, a super column family, or a column within a super column.

## System Architecture

(The system architecture also plagiarizes the Dynamo paper in several places. I'm not sure what's going on with this, but it makes the paper feel really unprofessional.)

At a high level, a read/write request can go to any node in the cluster. That node works out who are the relevant replicas. For a write, it forwards the write request to all the replicas and waits for a quorum to reply.

For a read, it depends what the client has asked for consistency-wise. If they are happy with a potentially stale read, the handling node will ask the closest replica for the value and forward that back to the client. If the client wanted something more consistent, then it will contact all replicas and wait for a quorum of responses.

Like Dynamo, it uses consistent hashing to determine owners for a key. This can result in uneven load distribution. Dynamo resolved this by introducing "virtual nodes", Cassandra instead has lightly loaded nodes move to a location in the ring where they would be more useful.

Replication is similar to Dynamo, at least in the simplest case. After the coordinator node for a key, the $N-1$ next nodes also store a copy of the data. There are more complicated configurations ("Rack Aware" and "Datacentre Aware") that will be more resilient to failures. In these cases, Cassandra uses ZooKeeper for leader election. In this case the leader will choose who should replica what, using knowledge of physical location of the machines. The data about who should replica what is stored in ZooKeeper and cached on each node.

Cassandra uses Scuttlebutt, a gossip system, to manage cluster membership. Failure detection is done with a complicated-sounding procedure called the $\Phi$ Accrual Failure Detector. This acknowledges that there might be errors in any given heartbeat message, and tries to build a model out of exchanged heartbeat responses to get an overall picture of whether a node really has died or not. I'd love to hear more details about why they went down this route, it feels very heavy-handed.

Like Dynamo, machines are only added to the cluster by a human operator. The mechanism is the same, with gossip and a couple of designated seed nodes. The new node will pick a token (determining its key range) at random, which will mean it streams a bunch of data off another node (at 40MB/sec, at the time of writing). (I imagine the human operator could also specify a token, if they wanted to specifically alleviate the load on a particular node.)

Every write a node receives is first persisted to the commit log, which is held on a dedicated disk. Then the update is reflected on an in-memory data structure. Once the in-memory data structure gets too large, it's flushed to disk along with an index that will allow fast lookups by row key. Over time these files are compacted.

A read first hits the in-memory structure, then goes to disk if necessary. There is a bloom filter for each file summarising which keys are present in it, so you can short-circuit what files you need to check. For wide rows, there is a column index (generated at every 256KB chunk).

The data structure and files are per-column family. There is a bit vector at the head of the commit log which can be used as a tracker for which column families have gone from memory to file. Once these are all set, you know the commit log is safe to delete. The actual switch for when the commit log is deleted is however its size (once it reaches 128MB).

It seems like the local persistence of Cassandra is really focussed on write throughput. They buffer data in memory and batch the writes to disk (including the commit log) so there will be data loss if the machine crashes.

There's a little bit more information about the data file and the index. The data file consists of blocks. Each block contains at most 128 keys, and is demarcated by a block index. The block index captures the relative offset of a key within the block and the size of its data. When the in-memory data structure is dumped to disk, the data is written and the index is generated. The index is also held in memory. (Presumably the bloom filter is also calculated and held in memory at this point.)
