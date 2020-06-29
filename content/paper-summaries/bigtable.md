---
title: "Bigtable: A Distributed Storage System for Structed Data"
date: 2020-06-09T09:53:35+01:00
---

I already looked at the GFS paper, which is just a distributed file store. Raw files might be fine for intermediate storage when you're just running a text processing MapReduce job, but most applications want a higher level of abstraction out of their storage mechanism. Enter [Bigtable](https://static.googleusercontent.com/media/research.google.com/en//archive/bigtable-osdi06.pdf).

Bigtable is (was?) a distributed storage system for structured data. They mention indexing, Google Earth, and Google Finance as using the system at the time. You can imagine that these applications require quite different things from the data store, which likely motivates their inclusion.

Bigtable is designed to scale to petabytes of storage and thousands of machines, which are similar numbers to what we saw for GFS. (This Bigtable paper is from 2006). As well as this scalability, it should also be performant and have high availability. They mention a few more applications at this point (some of which have sadly [passed away](https://killedbygoogle.com/) saying that some require through-put oriented batch processing whereas others require latency-sensitive real-time results for end users. (My experience as an end user of Bigtable has been that availability is indeed good, but it's easy to shoot yourself in the foot performance-wise.)

People weren't using the term NoSQL when this paper was written, so the authors say that it "in many ways resembles a database...[but] does not support a full relation model". We'll got more on what it _does_ provide to clients later on, but one thing to say for now that data is indexed using row and column names than can be arbitrary strings. Also, Bigtable treats all data as uninterpreted strings. Clients control data locality through their schema choice, through which they can also configure whether data is served from memory or disk.

## Data Model

A Bigtable is a sparse, distributed, persistent multi-dimensional sorted map. The map is indexed by row key, column key, and a timestamp. Each value in the map is an uninterpreted array of bytes. As an example, they give a Webtable, where the row keys are URLs, an example column key would be the page content, and they would have an entry of the actual page content at various timestamps.

Row keys are typically 10-100 bytes in size, though can technically be up to 64KB. All operations under a given row key are atomic. Bigtable stores data lexigraphically ordered by row key. The data is dynamically partitioned along the row key axis in to _tablets_, which is the unit of scaling and loadbalancing. This means that reads of sequential row keys are efficient (they will likely be reads from the same chunk, hence from the same machine) which the client can take in to consideration in schema design. As an example, in a web indexing application, you would likely want to use a reversed domain as a row key.

Column keys are grouped in to column families, which form the basic unit of access control. Column families have to be specified upfront, but once the family has been created any key can be used under that family. The naming convention is `familiy:qualifier`. Sometimes there is a single `qualifier` for a `family` (for example, language in a web page) and sometimes there are many (for example, a column family for referring pages).

Every entry has a timestamp. This can be assigned by Bigtable or by the client. Different versions of an entry are stored in decreasing timestamp order so that the most recent version can be read first. There is built in garbage collection for old timestamps (the client can specify e.g. to keep the last N timestamps, or to only kep values written in the last 7 days, etc.).

## API

The API has CRUD for tables and column families, as well as management for ACLs etc. There's some C++ code for updating a row using a `RowMutation` abstraction, it looks like the kind of code you would expect for a client updating a distributed map.

There's also an example of a `Scanner`. It's in the context of the webtable again, reading all of the "anchors" (and the associated timestamps) for a given URL. It looks like standard code for reading a stream. The client could also specify additional filters here (e.g. patterns the anchor should match).

Bigtable supports single-row transactions, but not multi-row transactions. It also offers the ability to make a cell an integer counter (the paper doesn't specify the exact API).

Definitely the weirdest bit of the paper so far: Clients can supply scripts written in Sawzall to be executed in the address space of servers. The Sawzall code isn't actually allowed to write back in to Bigtable, though.

## Building Blocks

Bigtable uses GFS under the hood to store log and data files. The Bigtable application itself is managed by the cluster management system (Borg, I guess) which handles the machine resource assignment etc.

They have a file format called SSTable, which provides a persistent ordered immutable map from keys to values (both keys and values being arbitrary bytestrings). There are access methods for reading values associated with a key and iterating over kv pairs in a key range. There's not loads of details on this file format, but they do mention that there's an index stored at the end of the file which is loaded in to memory to support these access methods.

Bigtable also uses Chubby extensively.

## Implementation

First off, there is a library that is linked in to clients. We saw some examples of the methods clients used from that library above.

Next, there is a master. The master is responsible for assigning tablets to tablet servers, adding/removing tablet servers, balancing load on the tablet servers, and garbage collection. It is also responsible for schema changes (table creation etc.) The master is _not_ responsible for tablet location information.

Finally, there are the tablet servers. Tablet servers will have somewhere between 10 and 10k tablets. The tablet server handles reads and writes to the tablets it owns, and it also splits tablets which have grown too large.

The heirarchy is a Bigtable cluster storing a number of tables, each table consisting of a number of tablets, and each tablet containg all data associated with a row range. Each table initially contains one tablet, and tablets are split off as they grow (target size 100-200MB by default).

Chubby stores the location of a file called the root tablet, which is the first tablet in a special `METADATA` table. The root tablet lives on some tablet server (servers, for durability) like the others, but is special in that it is never split. The `METADATA` table has rows whos keys are table identifiers. Obviously there are concerns about things getting hot here. Clients cache both the location of the root tablets and the tablets they need for their application. If they detect that their tablet information is out of date, they go back to the root tablet for an update, and if that's moved they go back to Chubby.

When tablet servers start up, they acquire a lock on a file in a specific directory in Chubby. The master monitors this directory to discover tablet servers and do the book-keeping for their health. The master handles tablet assignment/unassignment coming from table creation/deltion by actively sending requests to the tablet servers telling them to do stuff. The tablet servers themselves are responsible for tablet splitting/merging, and there's a bit more work to keep things up to date in this case.

This is the basic architecture, but there are quite a lot of tricks added in to make it more performant. I'm focussing on big picture here so won't summarise the section on "Refinements".

## Experiments

They did a few experiments filling up $N$ tablet servers for various values of $N$. They are filled with random data, with the payload size 1kB. They look at all combinations of random/sequential reads/writes, and also scanning.

Random reads are slow in their configuration because it involves sending a 64KB SSTable file over the network only to read 1kB from it. They say that an application needing performant random reads should lower the size of the SSTable files.

Sequential reads do better, but scans are better still. The improvements of scans vs sequential reads is just that you can cut out some unncessary RPCs in the scan case. (Naive sequential reads over a long range would need an RPC every time you finish a tablet).

Sequential and random writes are mostly comparable.

They throw extra machines at the problem and generally see things improving. Of course, the performance is not linear in the number of machines, because this is the real world.

## Closing Off

They talk about a few applications and some challengings of building the system. There's the usual ones about unexpected failure modes and assumptions made about other systems that they're integrating with. They mention the importance of not building stuff until you need it, and give multi-row transactions as an example. (This is surely a defensive statement, it's a controversial topic). They also talk about the value of simple design and not going too far off the beaten track in terms of using other systems.

Interestingly, they talk about the importance of monitoring and in particular how they used distributed tracing to understand the system. (It's not called distributed tracing at this point).
