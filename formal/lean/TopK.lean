set_option warn.classDefReducibility false

namespace TopK

class TotalOrd (α : Type) where
  le : α → α → Prop
  dec : DecidableRel le
  total : ∀ a b, le a b ∨ le b a
  trans : ∀ a b c, le a b → le b c → le a c
  antisymm : ∀ a b, le a b → le b a → a = b

attribute [instance] TotalOrd.dec

open TotalOrd

variable {α : Type} [TotalOrd α]

def ins (x : α) : List α → List α
  | [] => [x]
  | y :: l => if le x y then x :: y :: l else y :: ins x l

def insf (acc : List α) (x : α) : List α := ins x acc

def isort (l : List α) : List α := l.foldl insf []

def bstep (k : Nat) (acc : List α) (x : α) : List α := (ins x acc).take k

def bfold (k : Nat) (acc xs : List α) : List α := xs.foldl (bstep k) acc

def scanChunk (k : Nat) (p : α → Bool) (c : List α) : List α := bfold k [] (c.filter p)

def mergeLocals (k : Nat) (ls : List (List α)) : List α := ls.foldl (bfold k) []

def Sorted (l : List α) : Prop := l.Pairwise le

theorem take_ins_take (x : α) (a : List α) (k : Nat) :
    (ins x (a.take k)).take k = (ins x a).take k := by
  induction a generalizing k with
  | nil => simp
  | cons y a ih =>
    cases k with
    | zero => simp
    | succ k =>
      simp only [List.take_succ_cons, ins]
      split
      · cases k with
        | zero => simp
        | succ k => simp [List.take_take]
      · simp only [List.take_succ_cons, ih]

theorem bfold_take (k : Nat) (xs : List α) :
    ∀ a : List α, bfold k (a.take k) xs = (xs.foldl insf a).take k := by
  induction xs with
  | nil => intro a; simp [bfold]
  | cons x xs ih =>
    intro a
    simp only [bfold, List.foldl_cons] at *
    rw [show bstep k (a.take k) x = (ins x a).take k from take_ins_take x a k]
    exact ih (ins x a)

theorem bfold_nil (k : Nat) (xs : List α) : bfold k [] xs = (isort xs).take k := by
  have h := bfold_take k xs []
  simpa [isort] using h

theorem mem_ins (x z : α) (l : List α) : z ∈ ins x l ↔ z = x ∨ z ∈ l := by
  induction l with
  | nil => simp [ins]
  | cons y l ih =>
    simp only [ins]
    split
    · simp
    · simp only [List.mem_cons, ih]
      exact or_left_comm

theorem ins_perm (x : α) (l : List α) : (ins x l).Perm (x :: l) := by
  induction l with
  | nil => exact List.Perm.refl _
  | cons y l ih =>
    simp only [ins]
    split
    · exact List.Perm.refl _
    · exact (ih.cons y).trans (List.Perm.swap x y l)

theorem ins_sorted (x : α) (l : List α) (h : Sorted l) : Sorted (ins x l) := by
  induction l with
  | nil => simp [ins, Sorted]
  | cons y l ih =>
    simp only [Sorted, List.pairwise_cons] at h
    obtain ⟨hy, hl⟩ := h
    simp only [ins]
    split
    · rename_i hxy
      simp only [Sorted, List.pairwise_cons, List.mem_cons]
      refine ⟨?_, hy, hl⟩
      intro z hz
      rcases hz with rfl | hz
      · exact hxy
      · exact TotalOrd.trans _ _ _ hxy (hy z hz)
    · rename_i hxy
      have hyx : le y x := by
        rcases TotalOrd.total x y with h | h
        · exact absurd h hxy
        · exact h
      simp only [Sorted, List.pairwise_cons]
      refine ⟨?_, ih hl⟩
      intro z hz
      rcases (mem_ins x z l).mp hz with rfl | hz
      · exact hyx
      · exact hy z hz

theorem foldl_ins (l : List α) :
    ∀ a : List α, Sorted a → Sorted (l.foldl insf a) ∧ (l.foldl insf a).Perm (a ++ l) := by
  induction l with
  | nil => intro a h; simp [h]
  | cons x l ih =>
    intro a h
    simp only [List.foldl_cons]
    obtain ⟨hs, hp⟩ := ih (ins x a) (ins_sorted x a h)
    refine ⟨hs, hp.trans ?_⟩
    exact ((ins_perm x a).append_right l).trans List.perm_middle.symm

theorem isort_sorted (l : List α) : Sorted (isort l) := (foldl_ins l [] (by simp [Sorted])).1

theorem isort_perm (l : List α) : (isort l).Perm l := by
  have h := (foldl_ins l [] (by simp [Sorted])).2
  simp only [List.nil_append] at h
  exact h

theorem eq_of_perm_sorted :
    ∀ l₁ l₂ : List α, Sorted l₁ → Sorted l₂ → l₁.Perm l₂ → l₁ = l₂ := by
  intro l₁
  induction l₁ with
  | nil => intro l₂ _ _ p; exact p.symm.eq_nil.symm
  | cons a t₁ ih =>
    intro l₂ h₁ h₂ p
    cases l₂ with
    | nil => exact absurd p.eq_nil (by simp)
    | cons b t₂ =>
      simp only [Sorted, List.pairwise_cons] at h₁ h₂
      have hb : b ∈ a :: t₁ := p.symm.subset (by simp)
      have ha : a ∈ b :: t₂ := p.subset (by simp)
      have hab : a = b := by
        rcases List.mem_cons.mp hb with e | hb'
        · exact e.symm
        · rcases List.mem_cons.mp ha with e | ha'
          · exact e
          · exact TotalOrd.antisymm _ _ (h₁.1 b hb') (h₂.1 a ha')
      subst hab
      rw [ih t₂ h₁.2 h₂.2 (List.Perm.cons_inv p)]

theorem take_sorted (l : List α) (k : Nat) (h : Sorted l) : Sorted (l.take k) :=
  List.Pairwise.sublist (List.take_sublist k l) h

theorem merge_step (k : Nat) (s F : List α) (hs : Sorted s) :
    bfold k (s.take k) ((isort F).take k) = (F.foldl insf s).take k := by
  have hL : Sorted ((isort F).take k) := take_sorted _ k (isort_sorted F)
  rw [bfold_take]
  have e1 : ((isort F).take k).foldl insf s = s.foldl insf ((isort F).take k) := by
    obtain ⟨a1, p1⟩ := foldl_ins ((isort F).take k) s hs
    obtain ⟨a2, p2⟩ := foldl_ins s ((isort F).take k) hL
    exact eq_of_perm_sorted _ _ a1 a2 (p1.trans (List.perm_append_comm.trans p2.symm))
  rw [e1, ← bfold_take, List.take_take, Nat.min_self, bfold_take]
  obtain ⟨a1, p1⟩ := foldl_ins s (isort F) (isort_sorted F)
  obtain ⟨a2, p2⟩ := foldl_ins F s hs
  have e2 : s.foldl insf (isort F) = F.foldl insf s :=
    eq_of_perm_sorted _ _ a1 a2
      (p1.trans ((((isort_perm F).append_right s)).trans (List.perm_append_comm.trans p2.symm)))
  rw [e2]

theorem merge_general (k : Nat) (p : α → Bool) (ords : List (List α)) :
    ∀ s : List α, Sorted s →
      (ords.map (scanChunk k p)).foldl (bfold k) (s.take k)
        = ((ords.flatten.filter p).foldl insf s).take k := by
  induction ords with
  | nil => intro s _; simp
  | cons c rest ih =>
    intro s hs
    simp only [List.map_cons, List.foldl_cons, List.flatten_cons, List.filter_append, List.foldl_append]
    have hstep : bfold k (s.take k) (scanChunk k p c) = ((c.filter p).foldl insf s).take k := by
      simp only [scanChunk, bfold_nil]
      exact merge_step k s (c.filter p) hs
    rw [hstep]
    exact ih _ (foldl_ins _ s hs).1

theorem topk_correct (k : Nat) (p : α → Bool) (cs ord : List (List α)) (h : ord.Perm cs) :
    mergeLocals k (ord.map (scanChunk k p)) = (isort (cs.flatten.filter p)).take k := by
  have g := merge_general k p ord [] (by simp [Sorted])
  simp only [List.take_nil] at g
  simp only [mergeLocals, g]
  have e : isort (ord.flatten.filter p) = isort (cs.flatten.filter p) :=
    eq_of_perm_sorted _ _ (isort_sorted _) (isort_sorted _)
      ((isort_perm _).trans (((List.Perm.flatten h).filter p).trans (isort_perm _).symm))
  rw [← e]
  rfl

theorem topk_unique (k : Nat) (p : α → Bool) (cs : List (List α)) (r : List α)
    (hs : Sorted r) (hp : r.Perm (cs.flatten.filter p)) :
    (isort (cs.flatten.filter p)).take k = r.take k := by
  rw [eq_of_perm_sorted _ _ (isort_sorted _) hs ((isort_perm _).trans hp.symm)]

theorem topk_length (k : Nat) (p : α → Bool) (cs : List (List α)) :
    ((isort (cs.flatten.filter p)).take k).length = min k (cs.flatten.filter p).length := by
  rw [List.length_take, (isort_perm _).length_eq]

theorem topk_matches (k : Nat) (p : α → Bool) (cs : List (List α)) (z : α)
    (hz : z ∈ (isort (cs.flatten.filter p)).take k) : p z = true := by
  have h1 := List.mem_of_mem_take hz
  have h2 := (isort_perm _).subset h1
  exact (List.mem_filter.mp h2).2

theorem topk_sorted (k : Nat) (p : α → Bool) (cs : List (List α)) :
    Sorted ((isort (cs.flatten.filter p)).take k) := take_sorted _ k (isort_sorted _)


theorem fused_fold {β γ σ : Type} (step : β → σ → γ → σ) (rows : List γ) :
    ∀ (init : β → σ) (q : β),
      (rows.foldl (fun st r => fun q' => step q' (st q') r) init) q = rows.foldl (step q) (init q) := by
  induction rows with
  | nil => intro init q; rfl
  | cons r rows ih => intro init q; exact ih _ q

def guardedStep {γ : Type} (k : Nat) (f : γ → α) (p : α → Bool) (b : List α) (r : γ) : List α :=
  if p (f r) then bstep k b (f r) else b

theorem guarded_fold {γ : Type} (k : Nat) (f : γ → α) (p : α → Bool) (rows : List γ) :
    ∀ acc : List α, rows.foldl (guardedStep k f p) acc = bfold k acc ((rows.map f).filter p) := by
  induction rows with
  | nil => intro acc; rfl
  | cons r rows ih =>
    intro acc
    simp only [List.foldl_cons, List.map_cons, List.filter_cons, guardedStep]
    split <;> simp [bfold, List.foldl_cons, ih]

def batchScanChunk {β γ : Type} (k : Nat) (f : β → γ → α) (p : β → α → Bool) (c : List γ) : β → List α :=
  c.foldl (fun st r => fun q => guardedStep k (f q) (p q) (st q) r) (fun _ => [])

theorem batchScanChunk_eq {β γ : Type} (k : Nat) (f : β → γ → α) (p : β → α → Bool) (c : List γ) (q : β) :
    batchScanChunk k f p c q = scanChunk k (p q) (c.map (f q)) := by
  unfold batchScanChunk scanChunk
  rw [fused_fold (fun q b r => guardedStep k (f q) (p q) b r) c (fun _ => []) q, guarded_fold]

theorem batch_correct {β γ : Type} (k : Nat) (f : β → γ → α) (p : β → α → Bool)
    (cs ord : List (List γ)) (h : ord.Perm cs) (q : β) :
    mergeLocals k (ord.map (fun c => batchScanChunk k f p c q))
      = (isort ((cs.map (·.map (f q))).flatten.filter (p q))).take k := by
  have e : ord.map (fun c => batchScanChunk k f p c q) = (ord.map (·.map (f q))).map (scanChunk k (p q)) := by
    simp [List.map_map, Function.comp_def, batchScanChunk_eq]
  rw [e]
  exact topk_correct k (p q) _ _ (h.map _)

structure Cand where
  score : Int
  id : Nat
  deriving Repr, DecidableEq

def candLe (a b : Cand) : Prop := a.score < b.score ∨ (a.score = b.score ∧ a.id ≤ b.id)

instance : DecidableRel candLe := fun a b => by unfold candLe; infer_instance

instance : TotalOrd Cand where
  le := candLe
  dec := inferInstance
  total := by intro a b; unfold candLe; omega
  trans := by intro a b c; unfold candLe; omega
  antisymm := by
    intro ⟨s1, i1⟩ ⟨s2, i2⟩ h1 h2
    simp only [candLe] at h1 h2
    simp only [Cand.mk.injEq]
    omega

def demoRows : List Cand :=
  [⟨3, 0⟩, ⟨1, 1⟩, ⟨2, 2⟩, ⟨1, 3⟩, ⟨0, 4⟩, ⟨1, 5⟩, ⟨2, 6⟩]

def demoTag (c : Cand) : Bool := c.id % 2 == 1 || c.id == 4

#eval mergeLocals 3 ([[⟨1, 5⟩, ⟨2, 6⟩], [⟨3, 0⟩, ⟨1, 1⟩, ⟨2, 2⟩], [⟨1, 3⟩, ⟨0, 4⟩]].map (scanChunk 3 demoTag))
#eval (isort (demoRows.filter demoTag)).take 3


end TopK

#print axioms TopK.topk_correct
#print axioms TopK.topk_unique
#print axioms TopK.batch_correct
