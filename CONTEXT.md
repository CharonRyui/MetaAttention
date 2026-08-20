# Domain Glossary

## Stateful operator
A computation that transforms an explicit mutable mathematical state for each sequence and may expose that state for continuation. The term describes behavior, not a model family.

## Algorithm IR
The typed mathematical description of a stateful operator: inputs, state transition, and readouts. It describes roles and semantics without naming a model, hardware target, layout, or schedule.

## State
The per-sequence value carried from one token transition to the next. In the first language schema it is one named matrix with two distinct ordered Feature roles.

## Feature role
A named logical axis participating in state algebra. Role order is meaningful: it determines the direction of an axis action and is not inferred from physical tensor layout.

## Propagation
The ordered transformation applied to prior State before additive input-derived terms are added. Propagations execute in authored left-to-right order.

## Injection
An input-derived additive contribution to the next State. Its factors are typed by the Feature roles they carry; missing roles may broadcast, while shared roles must be combined explicitly.

## Readout
A named output derived from the post-transition State and current token inputs by one explicit State Contraction. Independent readouts may observe the same State.

## Head Mapping
The declared relationship between input Head roles and the State Head role. It defines both forward sharing and the inverse gradient reduction.

## Scanable Program
A mathematically valid Algorithm IR program whose per-token transition has an associative compact summary whose storage is independent of sequence length. Scanability is a semantic classification, separate from hardware feasibility.

## Recurrent Program
A mathematically valid program for which this compiler has no proven compact associative summary. It is rejected by the compiler rather than executed through a sequential fallback.

## ExecutionResult
The immutable public result of every Stateful Operator invocation. It contains named outputs and, only when requested, named final State; the result shape never changes with the number of outputs or states.

## Target capability
The concrete hardware, dtype, layout, and resource envelope accepted by a compiler specialization. A capability failure does not change a program's Scanable classification.

## Structural optimization
A compiler-selected equivalent representation or lowering justified by typed Algorithm IR structure and specialization metadata. It is shared by every matching program and never depends on a model name or complete known profile identity.
_Avoid_: Algorithm-specific optimization, profile-specific kernel, model dispatch
