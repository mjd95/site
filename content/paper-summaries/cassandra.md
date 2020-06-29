---
title: "Cassandra - A Decentralized Structured Storage System"
date: 2020-06-23T09:53:35+01:00
---

Today it's the [Cassandra paper](https://www.cs.cornell.edu/Projects/ladis2009/papers/Lakshman-ladis2009.PDF). This is a pretty old paper again (2009). Cassandra isn't too fashionable these days, but it's undeniably been a pretty big player in the last decade and deserves some attention.

So, it's 2009, you're working at Facebook. You're growing _very_ quickly, you have increasing storage needs day-by-day. You more or less know how to scale the application horizontally. Wouldn't it be nice if we could scale the database as well? Enter Cassandra.

Like the Google File System and basically anything actually usable, this is a system that needs to run on a large number of commodity servers, so it must be reslient to failure. Unlike the (raw) Google File System, Cassandra is a database. The term NoSQL was just emerging at the time, but in modern terminology we would call it a [Wide Column Store](https://en.wikipedia.org/wiki/Wide-column_store). Since this database is going to be used by the application serving user traffic, it will definitely need to be performant as well as reliable.

Cassandra was initially designed to solve the "Inbox Search" problem. Each user has an inbox of messages that has been sent to them, and the user would like to be able to search these. This would require the system to handle billions of writes per day, and the system should be able to scale up with new users. (In terms of user numbers, they had 100 million in 2008 at deployment time, and 250 million in 2009 when the paper was published. Of course this continued to grow exponentially for a long time.) Over time, Cassandra became the storage backend for various other services.

## Related Work

They mention GFS, which I wrote about recently. By the time this paper was written, the single-master aspect of GFS had been augmented with Chubby so it's fault-tolerant, but it's pretty much still the same design. In particular, the master is still the scalibility bottleneck. There's also BigTable built on top of this.

They also mention Dynamo (the original one, not DynamoDB) from Amazon. Their short summary suggests that Dynamo (originally designed for shopping carts) prefers availability over consistency, and reliance on timestamp reads (for conflict resolution) during writes affects write throughput.

There's also a bunch of other systems mentioned, which are less famous (at least in industry).

## Data Model

A table in Cassandra is a distributed, multi-dimensional map indexed by a key. The key is a string with no size contraints, but typically 16-36 bytes long. All operations on the values for a given key are atomic.

The values themselves are structured objects. Columns (the attributes appearing in the value) are grouped together in to sets called "column families" 
