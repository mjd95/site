---
title: "Container memory usage via the Docker API"
date: 2019-09-24T08:08:00+01:00
draft: true
---
One of the things Improbable does is provide a hosted game service.  At the most basica level, when a studio is hosting their game with us, they submit game server binaries which we run in docker containers on their behalf.  Obviously the studio cares about how much memory and CPU their game servers are using, both in development and when the game is live.  We had offered some insights along these lines in the past, but they were at a node level, which was missing the granularity that some customers wanted as they (rightly) bin-packed different worker containers onto the same node.

The standard way of analysing the CPU and memory usage of a collection of containers is to run containerd on the node, and we definitely considered this.  The downside of this approach, though, is that containerd exports a few thousand time series, whereas we only really care about a couple.  Our ingestion stack could drop all containerd metrics we don't care about, but even that involves processing time on the input.

Instead we decided to instrument the application directly.  We run a supervisor container ourselves on each node, which is responsible for starting and stopping the containers on that node, as well as interfacing with our orchestration API.  This supervisor could scrape the docker API and collect metrics for each container, and expose these to our metrics ingestion pipeline.
