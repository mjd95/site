---
title: "Dynamo: Amazon's Highly Available Key-Value Store"
date: 2020-06-16T09:53:35+01:00
draft: true
---

I was looking at the Bigtable paper last week and seems pretty natural to move on to the [Dynamo paper](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf) from there. This was written in 2007, a year after the Bigtable paper. My understanding of the problem space is that these two system make the same trade-off, sacrificing consistency in favour of availability. Sure, consistency is very important for certain applications, but for many others it is not, so it's worth sacrificing if there's a win to be made.

Something that's always been a bit confusing to me is the relationship between Dynamo and DynamoDB. According to [this blog post](https://www.allthingsdistributed.com/2012/01/amazon-dynamodb.html), it seems Dynamo did not gain much adoption internally due to operational complexity. DymanoDB is a successor taking some lessons from Dynamo (including the big one that no one likes a system that they have to operate / is difficult to operate) but it seems like there's more than just a rebrand as a managed service.

The application Dynamo was originally designed for is the shopping cart. They want this to be very highly available as an outage blocks sales and costs Amazon a lot of money. They can sacrifice a bit of consistency to achieve this, because a lot of sales are going to go through despite consistency problems with users fixing their carts manually.

At this point S3 is already available (internally and externally). S3 is also highly available and scalable, but is not very performant for certain application use cases. (Dynamo has a design goal of making the performance tradeoff against the other desiderata tuneable, but this seems to have hurt usability.)

Lots of applications only really need primary-key access to a datastore (shopping carts, session management, ...). You _could_ use a full blown relational database here, but there's a lot of overhead and it limits the scalability and availability. Dynamo will strip down the feature set to just a key-value store. It uses a bunch of standard tricks to achieve it, we'll cover those below. On a first glance, the system does indeed sound fairly complicated to operate (recall nightmares of debugging a gossip-based distributed system).

## System Design

Data items will be uniquely identified by a key, the value is a blob. Consistency requirements will be relaxed. It won't offer any isolation guarantees, and there are no multi-key transactions. They're thinking internal-only at this point, and they want internal engineers to be able to configure Dynamo to suit their latency and throughput requirements.

On the CAP side of things: It's all well to say that you prefer availability over consistency, but what does that actually mean? How do you handle conflicts? They say that specifically they want to have high availability for writes (i.e., reject the solution where the client can't write data if things are looking inconsistent) and will sacrifice read performance to achieve this. In terms of who resolves conflicts (data store or client application), the mechanisms available on the data store are going to be quite limited (last write wins), so they favour the client resolving conflicts. (Though the client can ask the data store to do it, if it really wants.)

Of course there is scalability as a design goal as well. They actively target a symmetric and decentralized architecture (so definitely no master), which they say simplifies the process of system scaling, provisioning, and maintenaning the system (which is debatable).

Finally there's some words on performance. They have latency sensitive applications in mind, and are targetting an SLA like 99.9% of read and write operations are performed within a few hundred milliseconds. This rules out having multiple hops on a request.

The interface offered to the client is just `get(key)` and `put(key, context, value)`. The `get` returns a `context` and the object / list of objects along with their version. The `put` decides where the data should go based on the key; the `context` is holding system metadata that the client doesn't really care about. (I expect there will be more information on this later on.)

The partitioning scheme for the keys is consistent hashing, with some tricks (there is a set of virtual nodes for each physical node) to distribute the load more evenly.

Each data item is replicated on $N$ physical hosts. The data item is first hashed on to a node (the _coordinator node_) and the coordinator replicates it to the next $N-1$ successor nodes in the ring. To actually achieve the durability, they might need to skip certain virtual nodes to ensure that these are the next $N-1$ _physical_ nodes (this is already feeling a bit ugly). The system is designed in such a way that every node knows the list of nodes storing a given key.
