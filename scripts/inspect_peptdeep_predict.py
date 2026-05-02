"""Inspect AlphaPeptDeep's predict_all internals so we can chunk correctly."""
import inspect
from peptdeep.protein.fasta import PredictSpecLibFasta

# 1. predict_all signature
print("=" * 60)
print("PredictSpecLibFasta.predict_all signature")
print("=" * 60)
try:
    print(inspect.signature(PredictSpecLibFasta.predict_all))
except Exception as e:
    print("predict_all not on class directly, looking up MRO ...")
    for klass in PredictSpecLibFasta.__mro__:
        if "predict_all" in klass.__dict__:
            print(f"  found in {klass.__name__}")
            print(inspect.signature(klass.predict_all))
            print(inspect.getsource(klass.predict_all)[:3000])
            break

# 2. Check for predict_mod_seqs / predict_one_batch helpers
print("\n" + "=" * 60)
print("All predict_* methods anywhere in MRO")
print("=" * 60)
seen = set()
for klass in PredictSpecLibFasta.__mro__:
    for name, _ in inspect.getmembers(klass):
        if name.startswith("predict") and name not in seen:
            seen.add(name)
            print(f"  {klass.__name__}.{name}")

# 3. ModelManager direct predict APIs that work on a precursor_df slice
print("\n" + "=" * 60)
print("ModelManager prediction APIs")
print("=" * 60)
from peptdeep.pretrained_models import ModelManager
for name, _ in inspect.getmembers(ModelManager):
    if name.startswith("predict") and not name.startswith("_"):
        print(f"  ModelManager.{name}")
