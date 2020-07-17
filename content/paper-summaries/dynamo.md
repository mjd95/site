---
title: "Dynamo: Amazon's Highly Available Key-Value Store"
date: 2020-06-16T09:53:35+01:00
---

I was looking at the Bigtable paper last week and seems pretty natural to move on to the [Dynamo paper](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf), written a year later, from there. Both of these systems operate in the same hyper-scale world, and relax/remove relational database concepts in order to achieve easier scalability. GFS had a fairly nuanced approach to consistency, but ultimately would reject writes in the case of a network partition. Dynamo is going to be more unapologetic about pushing resolution complexity to clients in order to be always writeable.

Something that's always been a bit confusing to me is the relationship between Dynamo and DynamoDB. According to [this blog post](https://www.allthingsdistributed.com/2012/01/amazon-dynamodb.html), it seems that Dynamo (the internal-only precursor) did not gain much adoption due to operational complexity. DymanoDB (the externally-available successor) has a similar design but learns some lessons from Dynamo (including that no one likes a system that is difficult to operate). But there's more than just a rebrand as a managed service between the two.

The application Dynamo was originally designed for is the shopping cart. The shopping card backend must be very highly available, as an outage blocks sales and costs Amazon a lot of money. Large scale is a given. They will pay some consistency costs for this. For the shopping cart, both adds and removals will be `puts` against the datastore. Reads can be stale, so the user might see an out of date version of their cart, but it's eventually consistent so all mutations to their cart should eventually be reflected.

Of course, the system also has to be performant. At this point S3 is already available (internally and externally). S3 is also highly available and scalable, but doesn't meet the performance requirements.

Then there is the data model. Lots of applications only really need primary-key access to a datastore (shopping carts, session management, and many more). You _could_ use a full blown relational database here, but there's a lot of overhead and it limits the scalability and availability. Dynamo will strip down the feature set to just a key-value store.

## System Design

Data items will be uniquely identified by a key, the value is a blob. Consistency requirements will be relaxed. It won't offer any isolation guarantees, and there are no multi-key transactions. They're thinking internal-only at this point, and they want internal engineers to be able to configure Dynamo to suit their latency and throughput requirements.

On the CAP side of things: It's all well to say that you prefer availability over consistency, but what does that actually mean? How do you handle conflicts? They say that specifically they want to have high availability for writes and will sacrifice read performance to achieve this. In terms of who resolves conflicts (data store or client application), the mechanisms available on the data store are going to be quite limited (last write wins), so they favour the client resolving conflicts. (Though the client can ask the data store to do it, if it really wants.)

Of course there is scalability as a design goal as well. They actively target a symmetric and decentralized architecture (so definitely no master), which they say simplifies the process of system scaling, provisioning, and maintenaning the system (which is debatable).

Finally there's some words on performance. They have latency sensitive applications in mind, and are targetting an SLA like 99.9% of read and write operations are performed within a few hundred milliseconds. This rules out having multiple hops on a request.

The interface offered to the client is just `get(key)` and `put(key, context, value)`. The `get` returns a `context` and the object / list of objects along with their version. The `put` decides where the data should go based on the key; the `context` is holding system metadata that the client doesn't really care about. (I expect there will be more information on this later on.)

The partitioning scheme for the keys is consistent hashing, with some tricks (there is a set of virtual nodes for each physical node) to distribute the load more evenly.

Each data item is replicated on $N$ physical hosts. The data item is first hashed on to a node (the _coordinator node_) and the coordinator replicates it to the next $N-1$ successor nodes in the ring. To actually achieve the durability, they might need to skip certain virtual nodes to ensure that these are the next $N-1$ _physical_ nodes (this is already feeling a bit ugly). The system is designed in such a way that every node knows the list of nodes storing a given key. This list is called the _preference list_.

We saw in the introduction that add/removes to the shopping cart are all `puts` that are eventually all reconciled in the client's view. Under the hood, Dynamo treats each modification as a new and immutable version of the data, and multiple versions of the object can exist at the same time. In the simplest case, versions of an object are serially ordered. In general though, object histories can get quite complicated and clients have to deal with that. For the shopping cart case, the eventual merge algorithm seems to be "definitely include all the adds, include all the removes that we're convinced of". (Unsurprising that they favour adds over removals...)

Dynamo uses vector clocks (basically a list of (node, counter) pairs) to keep track of object versions. A vector clock is given to every object version. When a client updates an object, it specifies which version it is updating (using the `context`), having obtained that version in a previous read request. If Dynamo was in an inconsistent state with multiple versions of the object, it would have given them all to the client. The client who does a put with multiple versions in their context is considered to have resolved that conflict. On the server side, nodes can use the vector clocks to determine whether the data item they are receiving is a descendant of what they know or not, and either bump a counter or extend the vector clock as appropriate.

Clients interested in reading/writing a key talk to Dynamo either through a generic load balancer (in which case they land up somewhere on the ring, and might get bounced if they're not on the preference list) or link in some Dynamo-specific code so they can land on the right node first time.

Consistency is tunable through two parameters: $R$, the minimum number of replicas that need to be involved in a successful read, and $W$, the same for writes. If $R+W \> N$ then there is a quorum-like system, but typically they have $R+W \< N$ because they are more worried about latency than consistency. For a `put`, the node receiving it stores it locally and forwards it to the $N$ highest-ranked reachable nodes, but only waits for $W-1$ to respond before acking the `put`. The read path is similar, with the coordinator returning all conflicting versions it got from the first $R$ nodes.

Nodes find that other nodes are temporarily unavailable during routine operation. Dynamo uses hinted handoff to work around temporary unavailability: if node A that should receive an update to some key is unavailable, the update will go to another node B, and the metadata will hint that it should have gone to node A. Node B will periodically check the health of node A again, and if it finds it healthy it will offload the data that it received on its behalf. This is another example of availability over consistency.

The system must also be able to fully synchronize replicas, e.g. to recover from a non-transient failure. To do this, each node maintains a Merkle tree for each key range it hosts. If two hosts want to check their view of a key range is in agreement, they exchange the roots of the Merkle tree. If they're not, they walk down the tree and work out what data they need to update/exchange to consolidate their views.

Nodes are added/removed to the Dynamo ring by a human at the command line. The addition of the new node will eventually be propagated via gossip. Partition and placement information also goes via gossip. There's a potential for a logical partition here if two nodes are added around the same time, but there are dedicated seed nodes in the ring that all nodes must talk to when joining the cluster, so this shouldn't take too long to reconcile. In terms of data, when a new node is added this means existing nodes will no longer be responsible for some of their data, and it will be transferred to the new node.

## Implementation

Nodes in Dynamo have three main responsibilities: a local persistence engine, request coordination, and membership and failure detection.

Code is written in Java.

The local persistence engine is pluggable based on the applications use cases. The most common on is Berkeley DB (embedded DB for key-value store).

The request coordination process constructs a state machine per request, which is responsible for managing the communications with all the other backend nodes (and orchestrating retries etc.)

There's not much information on the actual implementation of membership and failure detection, but it was discussed fairly extensively at a high level above.

## Experiences

They highlight three patterns for Dynamo use, differentiated based on how reconciliation is done:

* Business logic specific reconciliation. This is what we saw above for the shopping cart
* Timestamp based reconciliation. This is Dynamo saying "last write wins". This could be used for session storage.
* High performance read engine. This could be setting $R=1$ and $W=N$. This could be a system storing promoted items, which receives few writes but lots of reads and wants those to be highly performant.

We haven't spoken that much about durability, but suffice to say that setting $W$ low risks losing data in the case of node failure.

The most common setting is $(N, R, W) = (3, 2, 2)$.

Generally speaking, writes are slower because they always have to go to disk. Dynamo does offer an optimisation of storing an in-memory write buffer for applications that need lower latency writes, trading durability for performance. To somewhat alleviate the durability issue, the coordinator insists that one of the nodes must write to disk, but the $W$ nodes which respond allowing the write to be acked will presumably be the ones who only had to write to the in-memory buffer.

There's an interesting comment about moving the state machine in to the request coordinator in to the client. This lowers latency a fair bit, mostly due to removing a network hop. It does mean that the client has to maintain a fairly up to date version of the cluster information and pull in more Dynamo dependencies, but in certain situations this could be a worthwhile trade-off.

## Closing Off

The discussion closes off by showing how Dyanamo held a solid SLA internally for its initially two years. It also touts Dynamo's high tunability. As downsides to onboarding applications, they mainly focus on the application having to work out how they want conflicts to be resolved.

They also comment that, for clusters with thousands of nodes, the cost of maintaining the routing table affects scalability.
