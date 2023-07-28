# --------
FROM nbhosting/minimal-notebook

# NOTE
# we cannot use jupyter nbextension enable/disable
# because we provide our own nbconfig/notebook.json
# and that is where is stored the enabled/disabled extensions
RUN true \
 && pip install numpy pandas matplotlib scipy pillow seaborn scikit_learn \
 && pip install -U --pre https://planete.inria.fr/Thierry.Parmentelat/dropbox/jupytext-1.15.0.dev2-py3-none-any.whl \
 && pip install -U jupyterlab-courselevels \
 && pip install -U ipythontutor nbautoeval \
 && true
