---- MODULE DistKVCore ----
EXTENDS Naturals, FiniteSets, TLC

(***************************************************************************)
(* A small, code-derived TLA+ model of the DistKV state machine.            *)
(*                                                                         *)
(* Source of truth (current code, not the older design docs):               *)
(*   - src/distkv/master.rs        MasterState: worker liveness/epoch/used, *)
(*                                  object metadata, put_start/commit, route *)
(*   - src/distkv/worker.rs        WorkerStore keyed by (key, generation)   *)
(*   - src/distkv/registration.rs  register / heartbeat epoch discipline    *)
(*   - src/kv_cache/local.rs       LocalKvHandle committed index            *)
(*   - src/kv_cache/distkv_store.rs local-first / read-locality fast path   *)
(*                                                                         *)
(* Deliberate abstractions (kept faithful where it affects safety):         *)
(*   - The wall clock is abstracted away. master.rs decides liveness with   *)
(*     `status == Alive && now - last_heartbeat <= TIMEOUT`, re-checked on   *)
(*     every get_route. Only the *result* of that check matters for safety, *)
(*     so we model it as one boolean `alive` per worker that Register and    *)
(*     Heartbeat set TRUE and `SuspectDead` (a heartbeat timeout *or*        *)
(*     mark_worker_dead) sets FALSE. This removes `now`, `lastHeartbeat`,    *)
(*     `AdvanceTime`, Timeout, TTL and the resulting state explosion.        *)
(*   - The read lease is omitted: get_route re-validates on each call, so no *)
(*     safety property depends on the lease.                                 *)
(*   - ObjectState Failed/Removed are dropped: MasterState never enters them.*)
(*   - PutId (a UUID) is a monotonically allocated integer.                  *)
(*   - Object bytes are abstract Values; `committedHist` is ghost state that *)
(*     records, per (key, generation), the value the Master accepted.        *)
(*                                                                         *)
(* Key safety facts the model must allow (not bugs):                         *)
(*   - A physical Crash can wipe a worker's bytes before the Master notices  *)
(*     (alive still TRUE). A route may then point at an empty worker -> the   *)
(*     read is a clean MISS, never wrong bytes.                              *)
(*   - put_commit may succeed even if the worker bytes never landed, so a    *)
(*     route can resolve to a slot that holds NoVal: again, a clean miss.    *)
(***************************************************************************)

CONSTANTS
  Keys,      \* finite set of object keys
  Workers,   \* finite set of worker ids
  Values,    \* finite set of abstract byte payloads
  Sizes,     \* finite set of object sizes used by PutStart
  Capacity,  \* per-worker capacity bound (bytes)
  MaxGen,    \* max object generation explored by TLC
  MaxPutId,  \* max allocated PutId explored by TLC
  MaxEpoch   \* max worker epoch explored by TLC

NoWorker == "NoWorker"
NoPut    == 0
NoVal    == "NoVal"

Absent   == "Absent"
Writing  == "Writing"
Complete == "Complete"

States  == {Absent, Writing, Complete}
Gen     == 0..MaxGen
RealGen == 1..MaxGen
PutIds  == 1..MaxPutId
Epochs  == 0..MaxEpoch

ASSUME Keys # {}
ASSUME Workers # {}
ASSUME Values # {}
ASSUME Sizes \subseteq 1..Capacity /\ Sizes # {}
ASSUME MaxGen >= 1 /\ MaxPutId >= 1 /\ MaxEpoch >= 1
ASSUME NoWorker \notin Workers
ASSUME NoVal \notin Values

VARIABLES
  workers,        \* Master worker metadata: registered / epoch / used / alive
  objects,        \* Master object metadata
  putSeq,         \* next PutId to allocate
  inflight,       \* client shadow: the latest open PUT per key
  crashed,        \* physical worker process crashed (Master may not know yet)
  store,          \* WorkerStore bytes: [worker][key][gen] -> value | NoVal
  storeSize,      \* physical size at each WorkerStore slot
  workerUsed,     \* WorkerStore used_bytes
  localIndex,     \* LocalKvHandle committed index: [worker][key] -> gen | 0
  committedHist   \* ghost: [key][gen] -> value the Master accepted | NoVal

vars == << workers, objects, putSeq, inflight, crashed,
           store, storeSize, workerUsed, localIndex, committedHist >>

InitWorker  == [ registered |-> FALSE, epoch |-> 0, used |-> 0, alive |-> FALSE ]
InitObject  == [ state |-> Absent, gen |-> 0, putId |-> NoPut, size |-> 0,
                 worker |-> NoWorker, workerEpoch |-> 0, reserved |-> FALSE ]
InitInFlight == [ active |-> FALSE, putId |-> NoPut, gen |-> 0,
                  worker |-> NoWorker, value |-> NoVal, size |-> 0 ]

EmptyStore     == [k \in Keys |-> [g \in Gen |-> NoVal]]
EmptyStoreSize == [k \in Keys |-> [g \in Gen |-> 0]]
EmptyLocalIdx  == [k \in Keys |-> 0]
EmptyCommitHist == [g \in Gen |-> NoVal]

Init ==
  /\ workers       = [w \in Workers |-> InitWorker]
  /\ objects       = [k \in Keys |-> InitObject]
  /\ putSeq        = 1
  /\ inflight      = [k \in Keys |-> InitInFlight]
  /\ crashed       = [w \in Workers |-> FALSE]
  /\ store         = [w \in Workers |-> EmptyStore]
  /\ storeSize     = [w \in Workers |-> EmptyStoreSize]
  /\ workerUsed    = [w \in Workers |-> 0]
  /\ localIndex    = [w \in Workers |-> EmptyLocalIdx]
  /\ committedHist = [k \in Keys |-> EmptyCommitHist]

SatSub(a, b) == IF a >= b THEN a - b ELSE 0

(***************************************************************************)
(* Master-observed liveness and routing (master.rs::worker_is_alive /       *)
(* get_route). `alive` already folds in the heartbeat-timeout check.         *)
(***************************************************************************)

MasterAlive(w) == workers[w].registered /\ workers[w].alive

Usable(w, sz) == MasterAlive(w) /\ workers[w].used + sz <= Capacity

Route(k) ==
  IF /\ objects[k].state = Complete            \* I1 NoDirtyRead
     /\ objects[k].worker # NoWorker
     /\ MasterAlive(objects[k].worker)         \* I2 PlacementHealth
     /\ workers[objects[k].worker].epoch = objects[k].workerEpoch
  THEN [ present |-> TRUE, worker |-> objects[k].worker, gen |-> objects[k].gen ]
  ELSE [ present |-> FALSE, worker |-> NoWorker, gen |-> 0 ]

(***************************************************************************)
(* Master / registration actions                                            *)
(***************************************************************************)

\* register_worker: new -> epoch 1; re-register -> epoch+1, used reset, alive.
\* A (re-)registration is a fresh in-memory WorkerStore + local index.
Register(w) ==
  /\ workers[w].epoch < MaxEpoch
  /\ workers' = [workers EXCEPT ![w] =
        [ registered |-> TRUE, epoch |-> workers[w].epoch + 1,
          used |-> 0, alive |-> TRUE ]]
  /\ crashed'    = [crashed EXCEPT ![w] = FALSE]
  /\ store'      = [store EXCEPT ![w] = EmptyStore]
  /\ storeSize'  = [storeSize EXCEPT ![w] = EmptyStoreSize]
  /\ workerUsed' = [workerUsed EXCEPT ![w] = 0]
  /\ localIndex' = [localIndex EXCEPT ![w] = EmptyLocalIdx]
  /\ UNCHANGED << objects, putSeq, inflight, committedHist >>

\* heartbeat: only a live (non-crashed) worker refreshes its liveness.
Heartbeat(w) ==
  /\ workers[w].registered
  /\ ~crashed[w]
  /\ workers' = [workers EXCEPT ![w].alive = TRUE]
  /\ UNCHANGED << objects, putSeq, inflight, crashed,
                  store, storeSize, workerUsed, localIndex, committedHist >>

\* The Master stops trusting w: either a heartbeat timeout or mark_worker_dead.
SuspectDead(w) ==
  /\ workers[w].alive
  /\ workers' = [workers EXCEPT ![w].alive = FALSE]
  /\ UNCHANGED << objects, putSeq, inflight, crashed,
                  store, storeSize, workerUsed, localIndex, committedHist >>

\* put_start: release the prior generation's reservation, pick a usable worker
\* (preferred if usable, else any usable one), reserve, bump generation.
PutStart(k, sz, preferred) ==
  /\ putSeq <= MaxPutId
  /\ objects[k].gen < MaxGen
  /\ LET obj == objects[k]
         released ==
           IF obj.reserved /\ obj.worker # NoWorker
           THEN [workers EXCEPT ![obj.worker].used = SatSub(@, obj.size)]
           ELSE workers
         UsableR(w) == MasterAlive(w) /\ released[w].used + sz <= Capacity
         candidates ==
           IF preferred \in Workers /\ UsableR(preferred)
           THEN {preferred}
           ELSE {w \in Workers : UsableR(w)}
     IN
       /\ candidates # {}
       /\ \E chosen \in candidates:
          \E val \in Values:
            /\ workers' = [released EXCEPT ![chosen].used = @ + sz]
            /\ objects' = [objects EXCEPT ![k] =
                 [ state |-> Writing, gen |-> obj.gen + 1, putId |-> putSeq,
                   size |-> sz, worker |-> chosen,
                   workerEpoch |-> released[chosen].epoch, reserved |-> TRUE ]]
            /\ inflight' = [inflight EXCEPT ![k] =
                 [ active |-> TRUE, putId |-> putSeq, gen |-> obj.gen + 1,
                   worker |-> chosen, value |-> val, size |-> sz ]]
            /\ putSeq' = putSeq + 1
            /\ UNCHANGED << crashed, store, storeSize, workerUsed,
                            localIndex, committedHist >>

\* put_commit: I3 + I4 -- commit only a Writing object whose put_id matches.
PutCommit(k) ==
  /\ objects[k].state = Writing
  /\ inflight[k].active
  /\ objects[k].putId = inflight[k].putId
  /\ objects' = [objects EXCEPT ![k].state = Complete]
  /\ committedHist' = [committedHist EXCEPT ![k][objects[k].gen] = inflight[k].value]
  /\ UNCHANGED << workers, putSeq, inflight, crashed,
                  store, storeSize, workerUsed, localIndex >>

(***************************************************************************)
(* WorkerStore / data-path actions (worker.rs)                              *)
(***************************************************************************)

\* put_bytes for the in-flight (key, generation) on its chosen worker.
WorkerWrite(k) ==
  /\ inflight[k].active
  /\ LET w == inflight[k].worker
         g == inflight[k].gen
         sz == inflight[k].size
         oldSize == storeSize[w][k][g]
         newUsed == workerUsed[w] - oldSize + sz
     IN
       /\ g \in RealGen
       /\ ~crashed[w]
       /\ workerUsed[w] >= oldSize
       /\ newUsed <= Capacity
       /\ store'      = [store EXCEPT ![w][k][g] = inflight[k].value]
       /\ storeSize'  = [storeSize EXCEPT ![w][k][g] = sz]
       /\ workerUsed' = [workerUsed EXCEPT ![w] = newUsed]
       /\ UNCHANGED << workers, objects, putSeq, inflight,
                       crashed, localIndex, committedHist >>

\* delete_generation: drop bytes for one (key, generation) slot.
Evict(w, k, g) ==
  /\ g \in RealGen
  /\ store[w][k][g] # NoVal
  /\ workerUsed[w] >= storeSize[w][k][g]
  /\ store'      = [store EXCEPT ![w][k][g] = NoVal]
  /\ workerUsed' = [workerUsed EXCEPT ![w] = @ - storeSize[w][k][g]]
  /\ storeSize'  = [storeSize EXCEPT ![w][k][g] = 0]
  /\ UNCHANGED << workers, objects, putSeq, inflight,
                  crashed, localIndex, committedHist >>

\* A physical crash wipes bytes/index. The Master is NOT told (alive unchanged).
Crash(w) ==
  /\ ~crashed[w]
  /\ crashed'    = [crashed EXCEPT ![w] = TRUE]
  /\ store'      = [store EXCEPT ![w] = EmptyStore]
  /\ storeSize'  = [storeSize EXCEPT ![w] = EmptyStoreSize]
  /\ workerUsed' = [workerUsed EXCEPT ![w] = 0]
  /\ localIndex' = [localIndex EXCEPT ![w] = EmptyLocalIdx]
  /\ UNCHANGED << workers, objects, putSeq, inflight, committedHist >>

(***************************************************************************)
(* Co-located read-locality fast path (distkv_store.rs::get_object):         *)
(* on a route to our own worker that has the bytes, record the committed     *)
(* generation in the local index for future local-first hits.               *)
(***************************************************************************)

LocalMarkFromRoute(w, k) ==
  /\ LET r == Route(k) IN
       /\ r.present
       /\ r.worker = w
       /\ store[w][k][r.gen] # NoVal
       /\ localIndex' = [localIndex EXCEPT ![w][k] = r.gen]
  /\ UNCHANGED << workers, objects, putSeq, inflight, crashed,
                  store, storeSize, workerUsed, committedHist >>

Next ==
  \/ \E w \in Workers: Register(w)
  \/ \E w \in Workers: Heartbeat(w)
  \/ \E w \in Workers: SuspectDead(w)
  \/ \E k \in Keys: \E sz \in Sizes: \E p \in Workers \cup {NoWorker}: PutStart(k, sz, p)
  \/ \E k \in Keys: PutCommit(k)
  \/ \E k \in Keys: WorkerWrite(k)
  \/ \E w \in Workers: \E k \in Keys: \E g \in RealGen: Evict(w, k, g)
  \/ \E w \in Workers: Crash(w)
  \/ \E w \in Workers: \E k \in Keys: LocalMarkFromRoute(w, k)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Invariants                                                               *)
(***************************************************************************)

TypeOK ==
  /\ workers \in [Workers -> [ registered: BOOLEAN, epoch: Epochs,
                               used: 0..Capacity, alive: BOOLEAN ]]
  /\ objects \in [Keys -> [ state: States, gen: Gen, putId: PutIds \cup {NoPut},
                            size: 0..Capacity, worker: Workers \cup {NoWorker},
                            workerEpoch: Epochs, reserved: BOOLEAN ]]
  /\ putSeq \in 1..(MaxPutId + 1)
  /\ inflight \in [Keys -> [ active: BOOLEAN, putId: PutIds \cup {NoPut},
                             gen: Gen, worker: Workers \cup {NoWorker},
                             value: Values \cup {NoVal}, size: 0..Capacity ]]
  /\ crashed \in [Workers -> BOOLEAN]
  /\ store \in [Workers -> [Keys -> [Gen -> Values \cup {NoVal}]]]
  /\ storeSize \in [Workers -> [Keys -> [Gen -> 0..Capacity]]]
  /\ workerUsed \in [Workers -> 0..Capacity]
  /\ localIndex \in [Workers -> [Keys -> Gen]]
  /\ committedHist \in [Keys -> [Gen -> Values \cup {NoVal}]]

\* I6 CapacityAccounting: neither the Master's reservations nor the worker's
\* physical bytes ever exceed capacity.
MasterCapacityBound == \A w \in Workers: workers[w].used <= Capacity
WorkerCapacityBound == \A w \in Workers: workerUsed[w] <= Capacity

\* A slot holds bytes iff it accounts for a non-zero size.
SlotSizeConsistency ==
  \A w \in Workers: \A k \in Keys: \A g \in Gen:
    (store[w][k][g] = NoVal) <=> (storeSize[w][k][g] = 0)

\* A route, when present, points at the current Complete generation, and that
\* generation was genuinely committed.
RouteIsCommittedGeneration ==
  \A k \in Keys:
    LET r == Route(k) IN
      r.present =>
        /\ objects[k].state = Complete
        /\ r.gen = objects[k].gen
        /\ committedHist[k][r.gen] # NoVal

\* The foundational no-corruption fact: any bytes a worker holds for a
\* committed generation equal exactly the committed value. Stale GETs land on
\* NoVal (wrong generation / evicted / crashed) instead of reused bytes.
StoreAgreesWithCommit ==
  \A w \in Workers: \A k \in Keys: \A g \in Gen:
    (store[w][k][g] # NoVal /\ committedHist[k][g] # NoVal) =>
      store[w][k][g] = committedHist[k][g]

\* I1 NoDirtyRead end-to-end: a remote read along a present route returns
\* either the exact committed bytes or a clean miss -- never wrong bytes.
RouteReadExactOrMiss ==
  \A k \in Keys:
    LET r == Route(k) IN
      r.present =>
        \/ store[r.worker][k][r.gen] = NoVal
        \/ store[r.worker][k][r.gen] = committedHist[k][r.gen]

\* Local-first fast path is safe: a co-located committed-index hit only ever
\* returns bytes that were genuinely committed at that generation.
LocalHitWasCommitted ==
  \A w \in Workers: \A k \in Keys:
    LET g == localIndex[w][k] IN
      (g # 0 /\ store[w][k][g] # NoVal) =>
        /\ committedHist[k][g] # NoVal
        /\ store[w][k][g] = committedHist[k][g]

(***************************************************************************)
(* NOT an invariant for mutable keys, on purpose:                           *)
(*                                                                         *)
(*   LocalHitIsLatest ==                                                    *)
(*     \A w \in Workers: \A k \in Keys:                                    *)
(*       localIndex[w][k] # 0 => localIndex[w][k] = objects[k].gen          *)
(*                                                                         *)
(* get_object consults LocalKvHandle before the Master, so after a remote   *)
(* overwrite a co-located reader can serve an older *committed* generation. *)
(* Safe only if object keys are immutable / content-addressed -- which KV    *)
(* cache keys are. The model still guarantees the value is a real committed  *)
(* value (LocalHitWasCommitted), just not necessarily the newest.           *)
(***************************************************************************)

====
