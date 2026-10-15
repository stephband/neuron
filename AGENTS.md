# Agent Guidelines

## Code Style & Formatting

Always adhere to the style and formatting rules configured in `.editorconfig`
and `deno.json` (`fmt` and `lint`).

### .editorconfig

@[.editorconfig](.editorconfig)

### deno.json

@[deno.json](deno.json)

## Architectural Guidelines

- **Style**: prefer a functional style, use classes where stateful objects are
  necessary or more memory efficient.
- **Helpers**: Import helper functions from external `fn`, `dom`, `midi` and
  `sequence` libraries where appropriate. Shared libraries are also located
  locally in `~/Sites/` (e.g. `~/Sites/fn/modules/weak-cache.js` for `weakCache`,
  `~/Sites/midi/modules/note.js` for `toRootName`). Consult or fix bugs directly
  in these local sibling repositories when debugging upstream library issues
  rather than patching over them in downstream consumer projects.
- **Classes**: Classes are for creating stateful objects (and not for
  namespacing collections of functions). Where prudent classes should be given
  static `.of()` and/or `.from()` functions as a means of instantiation.
- **Module Exports**: Where the primary interface of a module is a single class
  or function export it as `default`. Where a module is collection of constants
  and/or functions export them as named exports. Prefer a one-module-per-class
  approach to classes. Do not re-export external dependencies or utilities from
  core modules and avoid `index.js` barrel modules. Consumers must import
  directly from origin.
- **File Naming**: File names should be dash-case of the camel case class or
  function name of their default export.
- **Prune Dead Code**: Never leave unused imports, unreferenced helper files, or
  orphan utilities in the tree.
