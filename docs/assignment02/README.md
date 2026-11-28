# Assignment 02 PPT

`../Assignment02_3D_motiondata.pptx` (4 slides) is built from the course template with these two scripts.

```bash
# 1. metrics + per-file diagnostics (ai_ml_final env, from final_project/)
python scripts/p1_3_analyze_postprocessing.py --no-plot > p13_log.txt
# 2. figures (ai_ml_final env, from final_project/)
python docs/assignment02/make_figures.py <figure_dir> p13_log.txt
# 3. slides (needs python-pptx: pip install python-pptx)
python docs/assignment02/build_pptx.py Assignment_3D_motiondata.pptx <figure_dir> docs/Assignment02_3D_motiondata.pptx
```

The template `Assignment_3D_motiondata.pptx` is the course file and is not stored here.
