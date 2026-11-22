namespace Wal

structure Row where
  present : Bool
  v : Nat
  live : Bool
  deriving DecidableEq, Repr

inductive Op where
  | append (id v : Nat)
  | del (id : Nat)
  deriving DecidableEq, Repr

def kill (r : Row) : Row := if r.present then { r with live := false } else r

def step (i : Nat) (r : Row) : Op → Row
  | .append j v => if j = i then ⟨true, v, true⟩ else r
  | .del j => if j = i then kill r else r

def applyOp (s : Nat → Row) (o : Op) : Nat → Row := fun i => step i (s i) o

def replay (s : Nat → Row) (L : List Op) : Nat → Row := L.foldl applyOp s

theorem replay_pointwise (L : List Op) : ∀ (s : Nat → Row) (i : Nat), replay s L i = L.foldl (step i) (s i) := by
  induction L with
  | nil => intro s i; rfl
  | cons o L ih => intro s i; exact ih (applyOp s o) i

inductive Eff where
  | id
  | kill
  | set (r : Row)

def Eff.run : Eff → Row → Row
  | .id, r => r
  | .kill, r => Wal.kill r
  | .set c, _ => c

def Eff.next (i : Nat) : Eff → Op → Eff
  | e, .append j v => if j = i then .set ⟨true, v, true⟩ else e
  | .id, .del j => if j = i then .kill else .id
  | .kill, .del _ => .kill
  | .set c, .del j => if j = i then .set (Wal.kill c) else .set c

theorem kill_kill (r : Row) : kill (kill r) = kill r := by
  unfold kill; split <;> simp_all

theorem next_spec (i : Nat) (e : Eff) (o : Op) (r : Row) : step i (e.run r) o = (e.next i o).run r := by
  cases o with
  | append j v => cases e <;> simp only [step, Eff.next, Eff.run] <;> split <;> rfl
  | del j =>
    cases e <;> simp only [step, Eff.next, Eff.run]
    · split <;> rfl
    · split <;> simp [kill_kill]
    · split <;> rfl

theorem fold_eff (i : Nat) (L : List Op) : ∀ (e : Eff) (r : Row), L.foldl (step i) (e.run r) = (L.foldl (Eff.next i) e).run r := by
  induction L with
  | nil => intro e r; rfl
  | cons o L ih => intro e r; simp only [List.foldl_cons, next_spec, ih]

theorem run_idem (e : Eff) (r : Row) : e.run (e.run r) = e.run r := by
  cases e <;> simp [Eff.run, kill_kill]

theorem fold_idem (i : Nat) (L : List Op) (r : Row) : L.foldl (step i) (L.foldl (step i) r) = L.foldl (step i) r := by
  have h : ∀ x, L.foldl (step i) x = (L.foldl (Eff.next i) Eff.id).run x := fun x => fold_eff i L Eff.id x
  rw [h (L.foldl (step i) r), h r, run_idem]

theorem prefix_idem (i : Nat) (A B : List Op) (r : Row) :
    (A ++ B).foldl (step i) (A.foldl (step i) r) = (A ++ B).foldl (step i) r := by
  simp only [List.foldl_append, fold_idem]

theorem replay_from_prefix_state (s m : Nat → Row) (W : List Op)
    (h : ∀ i, ∃ P, P <+: W ∧ m i = replay s P i) : replay m W = replay s W := by
  funext i
  obtain ⟨P, ⟨B, rfl⟩, hm⟩ := h i
  rw [replay_pointwise, replay_pointwise, hm, replay_pointwise, prefix_idem]

theorem overlay_idem (s : Nat → Row) (A B : List Op) : replay (replay s A) (A ++ B) = replay s (A ++ B) :=
  replay_from_prefix_state s _ _ (fun _ => ⟨A, List.prefix_append A B, rfl⟩)

theorem partial_checkpoint (s m : Nat → Row) (W1 W2 W3 : List Op)
    (h : ∀ i, m i = replay s W1 i ∨ m i = replay s (W1 ++ W2) i) :
    replay m (W1 ++ W2 ++ W3) = replay s (W1 ++ W2 ++ W3) := by
  refine replay_from_prefix_state s m _ (fun i => ?_)
  rcases h i with h | h
  · exact ⟨W1, ⟨W2 ++ W3, by simp⟩, h⟩
  · exact ⟨W1 ++ W2, List.prefix_append _ _, h⟩

def empty : Nat → Row := fun _ => ⟨false, 0, false⟩

theorem stale_prefix_unsafe :
    replay (replay empty [.append 0 1, .del 0]) [.append 0 1] 0 ≠ replay empty [.append 0 1, .del 0] 0 := by
  decide

structure Frame where
  op : Op
  commit : Bool
  deriving DecidableEq, Repr

def validPrefix : List (Option Frame) → List Frame
  | [] => []
  | none :: _ => []
  | some f :: r => f :: validPrefix r

def commitLen : List Frame → Nat
  | [] => 0
  | f :: r => if commitLen r > 0 then commitLen r + 1 else if f.commit then 1 else 0

def recover (disk : List (Option Frame)) : List Frame := (validPrefix disk).take (commitLen (validPrefix disk))

def Agrees (disk : List (Option Frame)) (written : List Frame) : Prop :=
  ∀ (k : Nat) (f : Frame), disk[k]? = some (some f) → written[k]? = some f

def EndsCommitted (L : List Frame) : Prop := ∀ f, L.getLast? = some f → f.commit = true

theorem validPrefix_prefix : ∀ (disk : List (Option Frame)) (written : List Frame),
    Agrees disk written → validPrefix disk <+: written := by
  intro disk
  induction disk with
  | nil => intro w _; exact List.nil_prefix
  | cons d disk ih =>
    intro w h
    cases d with
    | none => exact List.nil_prefix
    | some f =>
      cases w with
      | nil => have := h 0 f rfl; simp at this
      | cons g w =>
        have e : g = f := by have := h 0 f rfl; simpa using this
        subst e
        have hw : Agrees disk w := fun k f' hk => by simpa using h (k + 1) f' (by simpa using hk)
        exact (List.prefix_cons_inj g).mpr (ih w hw)

theorem commitLen_ends : ∀ L : List Frame, EndsCommitted (L.take (commitLen L)) := by
  intro L
  induction L with
  | nil => intro f h; simp [commitLen] at h
  | cons g r ih =>
    intro f h
    simp only [commitLen] at h
    split at h
    · rename_i hpos
      have hp : r.take (commitLen r) ≠ [] := by
        intro hnil
        rcases List.take_eq_nil_iff.mp hnil with h0 | h0
        · omega
        · subst h0; simp [commitLen] at hpos
      rw [List.take_succ_cons, List.getLast?_cons] at h
      cases hx : (r.take (commitLen r)).getLast? with
      | none => exact absurd (List.getLast?_eq_none_iff.mp hx) hp
      | some x =>
        rw [hx] at h
        simp only [Option.getD_some, Option.some.injEq] at h
        subst h
        exact ih x hx
    · split at h
      · simp only [List.take_succ_cons, List.take_zero, List.getLast?_singleton, Option.some.injEq] at h
        subst h; assumption
      · simp at h

theorem recover_atomic (disk : List (Option Frame)) (written : List Frame) (h : Agrees disk written) :
    recover disk <+: written ∧ EndsCommitted (recover disk) :=
  ⟨(List.take_prefix _ _).trans (validPrefix_prefix disk written h), commitLen_ends _⟩

theorem validPrefix_some_append : ∀ (L : List Frame) (rest : List (Option Frame)),
    validPrefix (L.map some ++ rest) = L ++ validPrefix rest := by
  intro L
  induction L with
  | nil => intro rest; rfl
  | cons f L ih => intro rest; simp [validPrefix, ih]

theorem commitLen_append_ge : ∀ (P X : List Frame), P ≠ [] → EndsCommitted P → P.length ≤ commitLen (P ++ X) := by
  intro P
  induction P with
  | nil => intro X h; exact absurd rfl h
  | cons f P ih =>
    intro X _ hend
    simp only [List.cons_append, commitLen, List.length_cons]
    cases P with
    | nil =>
      have hf : f.commit = true := hend f (by simp)
      simp only [List.nil_append, List.length_nil, hf, ite_true]
      split <;> omega
    | cons g P =>
      have hP : EndsCommitted (g :: P) := fun x hx => hend x (by
        rw [List.getLast?_cons, hx]; rfl)
      have := ih X (by simp) hP
      simp only [List.length_cons] at this ⊢
      split <;> omega

theorem recover_durable (written : List Frame) (rest : List (Option Frame)) (c : Nat)
    (hc : c ≤ written.length) (hcommit : EndsCommitted (written.take c)) :
    c ≤ (recover ((written.take c).map some ++ rest)).length := by
  simp only [recover, validPrefix_some_append, List.length_take, List.length_append]
  rcases Nat.eq_zero_or_pos c with h0 | hpos
  · omega
  · have hne : written.take c ≠ [] := by
      intro hnil
      rcases List.take_eq_nil_iff.mp hnil with h | h
      · omega
      · subst h; simp at hc; omega
    have := commitLen_append_ge _ (validPrefix rest) hne hcommit
    simp only [List.length_take] at this
    omega

#print axioms replay_from_prefix_state
#print axioms partial_checkpoint
#print axioms recover_atomic
#print axioms recover_durable

end Wal
