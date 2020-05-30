---
title: "Tuesday Paper: In Search of an Understandable Consensus Algorithm"
date: 2020-05-26T08:53:00+01:00
---

First up is the [Raft paper](https://raft.github.io/raft.pdf). Raft is a consensus algorithm, which is to say that you have a collection of servers that you would like to all compute the same state machine. This is obviously useful for fault tolerance in distributed systems.

Paxos is a well-known consensus algorithm. Everything I've read about Paxos talks about it being difficult to understand, and this paper is no different. In fact, inspired by the Paxos issue, _understandability_ is a first-class goal of their algorithm design.

At the time of the paper (2014), real-world examples of replicated state machines included Chubby (Paxos) and ZooKeeper (ZAB). Now we would also add ETCD (Raft) to that list, among other things.

In consensus problems, the replicated state machines are normally implemented with a replicated log, so at least part of the task is keeping a consistent replicated log. We imagine any server in the collection receiving a state update from some client which it will add to its own log and replicate that log entry to the other servers. Once that leader server is happy that the log entry is agreed upon by all the servers, it will update its own state machine and return the result to the client.

There are certain properties that are required / desired for any practical system:

  * _Safety_: The system never returns an incorrect result under all non-Byzantine conditions
  * _Available_: The system is functional as long as a majority of servers are operational and can communicate with each other (and clients)
  * The system does not depend on timing to ensure the consistency of the logs (because here be dragons)
  * A minority of slow servers does not impact overall slow system performance

## What's Wrong With Paxos?

The authors split Paxos up in to _single-decree Paxos_ (a consensus algorithm for agreeing on a single replicated log entry) and _multi Paxos_ (a way of combining single-decree Paxos decisions to agree on a full replicated log).

Paxos is apparently difficult, even for researchers. Even the simplest part, single-decree Paxos, is difficult. Moreover, the version you want for applications (multi Paxos) is only really sketched in the paper, so there is not really consensus (hah) on a single algorithm for implementing the full replicated log.

## The Raft Algorithm

Raft elects a leader, who is then responsible for managing the replicated log. The leader accepts log entries from clients, replicates them to other servers, and tells the other servers when it is safe to apply the log entries to their state machines. Leader election is therefore a first class concept in Raft, which we'll discuss in detail.

First some basics. A Raft cluster is a collection of servers (typically five, which could survive two failures), and any server is either a _leader_, a _follower_, or a _candidate_. In normal operations, there is one _leader_ responsible for all communication with clients, and the remaining servers are _follower_. The _candidate_ stage only comes up during leader election.

Raft divides time in to _terms_ (of arbitrary length), each term being numbered with consecutive integers. Terms start with an election. If a leader is successfully elected, then the there will be normal operation for that term. If no leader is successfully elected, then there won't be normal operation this term, but a new one will be declared shortly.

Each server stores its own _current term number_, which may well become out of date. Current term numbers are exchanged every time the servers communicate, and servers will update to the newer value if they see their own is out of date. If a candidate or leader is the one who is out of date, it immediately makes itself a follower. On the flip side of that, if a follower receives a request to update the log with an entry with an old term number, it will reject it. I don't think the paper says exactly what ensues with this "outdated leader" scenario, but that outdated leader will know that another election has occurred since its own claim so it should become a follower.

There are only two RPCs in Raft: `RequestVote` (initiated by candidates during an election) and `AppendEntries` (initiated by leaders to replicate log entries and to provide a heartbeat).

### Leader Election

Servers start as followers, and remain followers as long as they receive valid RPCs from a leader. Leaders send out empty `AppendEntries` if necessary to maintain their authority.

If a follower receives nothing within the _election timeout_, it starts an election by incrementing its current term and becoming a candidate. It votes for itself and issues `RequestVote`s to all other servers in the cluster. There are now three possiblities:

  * The candidate wins the election. This happens as soon as it gets a majority of votes. In this case it becomes the leader.
  * The candidate receives a heartbeat from another server claiming to be the leader. If the heartbeat's term index is greater than or equal to the client's local version, then our candidate respects the sender as the new leader and becomes a follower. If the hearbeat's term index is smaller, then our candidate rejects this and continues as candidate.
  * There is no winner within the timeout. (For example, there is a split vote). In this case, a new term is created and the process repeats. (Without intervention, this could repeat indefinitely.)

Each client chooses how long it will wait for an election at random. So even if we were going to hit the inconclusive stage this round, one of the candidates would become frustrated with that inconclusion first, get its requests out early, and would most likely become the leader next round. 

### Log Replication

The leader receives a new log entry from a client. It appends it to its own local log and then issues an `AppendEntries` call to all other cluster members. Once it is _committed_ (defined immenently) the leader actually exectues the statement from its own log to update its state machine, and then responds to the client.

The leader decides that a log entry is committed as soon as it is replicated to a majority of servers. The leader is the authority for what has been committed and it tracks this with a high watermark index; it broadcasts this committed index widely (e.g. in heartbeats). That way, if a follower has received a log entry but does not yet know it's committed, it might learn so in a future message. Once it becomes committed, the follower updates its own state machine. The leader also updates its own state machine as new messages become committed.

The term and index are unique IDs for the log entries. It's absolutely crucial that if two logs have an entry with the same ID, then they are equal up to that point. So to ensure no entries are missed, the leader sends the previous ID along with the current entry, and if the previous ID does not match the follower's head then the append is rejected.

If a follower is partitioned from the leader for a while then it can fall behind. Even worse, if a leader crashes with some uncomitted logs, then the next _leader_ will be behind. A leader will never remove entries from its own logs, but it has no qualms about doing so to other logs. If it finds a follower who is ahead of its own log, it will tell that follower to delete those extra entries. If it finds a follower who is behind, the consistency check will fail and the leader will walk back through its own log until it finds something the client can accept.

There must be a caveat here though. The leader cannot ask a follower to delete a log entry that has already been committed. This will be discussed in the next section.

### Safety

Well, this is pretty simple fix to ensure the caveat from the previous section: Raft imposes that only a server which has all of the committed entries from the previous terms is eligble to become leader.

Recall that a server must get a majority of votes in order to become leader. Every committed message is on at least one server in that majority. So if a candidate advertises the latest entry in its own log, and every voting server checks that that log contains all the entries that the voter knows to be committed and only approves if so, then we can guarantee that only candidates which have all committed logs can become leader.

There are situations in which a leader might think it should be responsible for finishing the committment of a log entry from a previous term. Raft does not do this directly. Rather leaders are only directly responsible for committing logs in their own term, and rely on the fact that appends have to be consistent to commit things from previous terms. Of course, this could mean that the entries from the previous term eventually get rolled back rather than committed, so this is very much a design choice.

### Timing and Availability

In order to actually progress the state machine, Raft needs a steady leader. There are pathological situations in which this would not happen. For a mostly healthy system, you need the broadcast time (time it takes a server to round-trip to another server in the cluster) to be significantly smaller than the election timeout (the (average) time servers will wait during a leadership election), which in turn should be significantly smaller than the mean time between failures for the servers. As an operator, you can tweak the election timeout. In the real world, there's no concerns about lining these up correctly.

## Conclusion

There's a fair bit more in this paper (cluster membership changes, log compaction, implementation and evaluation) but I've spent my allotted hour on this so I'm just going to jump forward to the related work and conclusion. Obviously Paxos is related, the main difference between it and Raft is that Raft puts leadership election centre-stage. There is also ZAB (ZooKeeper) and Viewstamped Replication, which like Raft have strong leaders but unlike Raft do not have a strict "leader to follower" flow of log entries.

The paper concludes with discussing the value of understandability and I think I agree. Although I skimmed it in places, I think I got a pretty solid view of how the Raft algorithm works from this and am reasonably convinced of its correctness from this reasonably short paper.
