# Tutorial

The tutorial has two parts, each a session that can be followed from start to end and that the test suite runs as written.

[Using plantbench](tutorial/using.md) goes from installation to a scored task on `jacketed_cstr`: running a case, changing its configuration, linear analysis, generating a dataset, its provenance, sensitivity indices and tasks.

[Contributing a case](tutorial/contributing.md) takes a case from the template to the library: a package of its own, a change to the plant, the contract and the library's rules, and `plantbench contribute`, which moves the case into a clone of the library. Its last section undoes every step, so it can be followed in a clone to learn the procedure without changing the library.

```{toctree}
:hidden:

tutorial/using
tutorial/contributing
```
