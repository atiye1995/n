# vqe_uq2 — خط لوله‌ی اصلاح‌شده برای ZNE با کمّی‌سازی عدم‌قطعیت

هیچ داده‌ی ساختگی‌ای در این بسته وجود ندارد. همه‌ی اعداد از این‌ها می‌آیند:
هامیلتونی واقعی (PySCF) ← VQE بدون نویز ← مدار گیت‌محور ← مدل نویز ساخته‌شده از **کالیبراسیون واقعی دستگاه IBM** ← نمونه‌برداری شات.
`tests/` فقط اتحادهای ریاضی با جواب تحلیلی را چک می‌کند و چیزی در `results/` نمی‌نویسد.

## نصب
```
pip install -r requirements.txt
python calibration.py --fake FakeSherbrooke --out data/calibration.json   # کالیبراسیون واقعی (snapshot دستگاه)
# یا برای دستگاه زنده: python calibration.py --backend ibm_xxx --qubits 0 1 2 3
python -m pytest tests -q
```
داده‌ی دیگری برای دانلود لازم نیست؛ فقط `data/calibration.json` را یک بار می‌سازی.

## اجرا
```
./run_all.sh H2 uccsd        # یا: python run_experiment.py --molecule LiH --ansatz adapt --shots 1000000
```
| فایل | کار |
|---|---|
| `chem.py` | هامیلتونی واقعی + انرژی دقیق **درون فضای فعال** (مرجع کیفیت آنسانز) |
| `ansatz.py` | UCCSD (تک+دوگانه) و ADAPT-VQE با گرادیان تحلیلی (adjoint) و L-BFGS |
| `circuits.py` | کامپایل دقیق ژنراتورها به گیت، ساده‌سازی، **folding واقعی** |
| `noise.py`, `dm_sim.py` | نویز گیت‌محور (T1/T2 + دپولاریزه) و شبیه‌ساز ماتریس چگالی |
| `measurement.py` | گروه‌بندی QWC، شات واقعی، خطای readout، readout mitigation |
| `estimators.py` | raw, Richardson, poly, exp, exp3, stacked (CV) + CRPS و interval score |
| `hier_model.py` | **روش پیشنهادی**: ZNE ساختاریافته‌ی سلسله‌مراتبی بیزی (NumPyro) |
| `analyze.py` | پوشش، RMSE، bias، interval score، خطای بی‌نهایت-شات |
| `noise_sweep.py` | آزمون کمّی نظریه: خطا بر حسب شدت نویز |
| `curvature.py` | جایگزین spike-and-slab قدیمی (آزمون انحنا روی داده‌ی شاتی، FDR) |
| `hardware_run.py` | همان تحلیل روی سخت‌افزار/fake backend (اجرا نشده؛ اول dry-run) |

## هر ایراد چگونه رفع شد
1. **مرجع**: همه‌ی خطاها نسبت به `E_ref` = انرژی VQE بدون نویز. خطای آنسانز جدا گزارش می‌شود.
2. **λ≥1**: folding واقعی گیت‌ها؛ محور برون‌یابی `λ_eff` است که از مدل نویز محاسبه می‌شود.
3. **درجه‌ی آزادی صفر / V_prior**: ≥۵ نقطه؛ میانگین‌گیری مدل با اعتبارسنجی برون‌یابی (نه evidence لاپلاس).
4. **نویز شات**: نمونه‌برداری واقعی از توزیع نتایج؛ σ از خود نمونه‌ها برآورد می‌شود (نه فرضی).
5. **np.trapz**: کد حذف شد؛ `curvature.py` بدون آن است.
6. **آماره‌ی z**: با واریانس شاتیِ واقعی، null تجربی و BH.
7. **آنسانز**: UCCSD کامل و ADAPT، گرادیان تحلیلی.
8. **نویز**: روی هر گیت، نه فقط حالت نهایی؛ `E(λ)` از folding واقعی می‌آید.
9. **خط پایه**: Richardson، poly، exp (سه‌نقطه‌ای و کمترین‌مربعات)، stacked، به‌علاوه‌ی کراس‌چک اختیاری با Mitiq.
- تخصیص شات: `--group-alloc {uniform,l1,oracle}` و `--lambda-alloc neyman`.

## محدودیت‌هایی که باید در مقاله بنویسی
- مدل نویز «device-informed» است، نه شبیه‌سازی دقیق دستگاه: اتصال کامل، بدون نویز idle/crosstalk، میانه‌ی پارامترها. برای ادعای واقعی حتماً `hardware_run.py` را اجرا کن.
- `oracle` واریانس دقیق را می‌داند؛ در عمل باید با شات آزمایشی برآورد شود.
- مدل سلسله‌مراتبی همبستگی درون گروه را در likelihood دارد اما بین گروه‌ها و بین λها مستقل فرض می‌کند؛ کالیبراسیون واقعی‌اش را `analyze.py` با پوشش تجربی می‌سنجد. اگر پوشش کم بود، همان را گزارش کن.
- چگالی-ماتریس تا ~۱۰ کیوبیت (H2, HeH+, H4, LiH). BeH2/N2/H2O فقط برای مطالعه‌ی آنسانز بدون نویز مناسب‌اند.
- CDR/PEC/symmetry-verification پیاده نشده‌اند.
- ناآزموده در محیط من: `chem.py`, `calibration.py`, `hier_model.py`, `hardware_run.py` (به pyscf/qiskit/jax نیاز دارند). بخش‌های numpy-محور تست شده‌اند.
