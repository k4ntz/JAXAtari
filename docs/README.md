# Documentation Guide (Sphinx + reST)

This project uses [Sphinx](https://www.sphinx-doc.org) for documentation, written in [reStructuredText](https://docutils.sourceforge.io/rst.html) (`.rst`)  and Markdown (`.md`) files.

## Getting Started

To build or extend the docs:

```bash
# 1. Install dependencies (e.g. inside a virtual environment)
pip install -r docs/source/requirements.txt

# 2. Navigate to the docs directory
cd docs/

# 3. Build HTML docs
make html

# Output will be in: docs/_build/html/index.html
```

### Auto-Doc Python Code
```bash
sphinx-apidoc -o docs/source/<output_path> jaxtari <module to document>
```

## Adding or Editing Docs

- All `.rst` files live in the `docs/` folder.
- The main entry point is `index.rst`, which includes other pages via the `toctree` directive.
- To add a new doc page:
  1. Create a new `mypage.rst`.
  2. Link it in `index.rst` under `.. toctree::`.

## Useful reST Features

You can make the docs more engaging and structured with these tools:

### Containers

Create two-column layouts or style blocks:

```rst
.. container:: twocol

   .. container:: leftside

      Your text here...

   .. container:: rightside

      |img|

.. |img| image:: _static/example.png
   :width: 200
   :alt: Example image
```

### Images

```rst
.. image:: _static/diagram.png
   :width: 400
   :alt: Architecture diagram
```

Always put image files in the _static directory.

### Code Blocks

```rst
.. code-block:: python

   def hello():
       print("Hello, Sphinx!")
```

### Cross-Referencing

```rst
See :ref:`usage-guide` for more info.
```

## Contributing

1. Follow the structure in existing `.rst` files.
2. Preview your changes with `make html`.
3. Submit a pull request!

---
