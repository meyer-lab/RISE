API Reference
=============

Core Functions
--------------

Factorization
~~~~~~~~~~~~~

.. automodule:: RISE.factorization
   :members: pf2, rise_pca_r2x, correct_conditions
   :undoc-members:
   :show-inheritance:

Data Import
~~~~~~~~~~~

.. automodule:: RISE.imports
   :members: import_thomson, import_lupus
   :undoc-members:
   :show-inheritance:

Preprocessing
~~~~~~~~~~~~~

.. autofunction:: parafac2.normalize.prepare_dataset

Visualization Functions
-----------------------

General Plotting
~~~~~~~~~~~~~~~~

.. automodule:: RISE.figures.commonFuncs.plotGeneral
   :members: plot_r2x
   :undoc-members:
   :show-inheritance:

Factor Plotting
~~~~~~~~~~~~~~~

.. automodule:: RISE.figures.commonFuncs.plotFactors
   :members: plot_condition_factors, plot_eigenstate_factors, plot_gene_factors
   :undoc-members:
   :show-inheritance:

PaCMAP Visualization
~~~~~~~~~~~~~~~~~~~~

.. automodule:: RISE.figures.commonFuncs.plotPaCMAP
   :members: plot_labels_pacmap, plot_gene_pacmap, plot_wp_pacmap
   :undoc-members:
   :show-inheritance:

Factor Stability
~~~~~~~~~~~~~~~~

.. automodule:: RISE.figures.figureS4
   :members: plot_fms_diff_ranks, calculateFMS
   :undoc-members:
   :show-inheritance:

Additional Modules
------------------

Logistic Regression
~~~~~~~~~~~~~~~~~~~

.. automodule:: RISE.logisticReg
   :members:
   :undoc-members:
   :show-inheritance:

Cell Gating
~~~~~~~~~~~

.. automodule:: RISE.gating
   :members:
   :undoc-members:
   :show-inheritance:

