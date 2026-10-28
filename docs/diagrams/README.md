# Diagrams

The sources are the `.excalidraw` files: open and edit them on excalidraw.com, in the Virgil
font. Re-render the SVG after an edit; the fonts are embedded, so GitHub shows them:

```bash
npx -y mcp-excalidraw-server@2.1.2 render docs/diagrams/post-training-loop.excalidraw \
    --out docs/diagrams/post-training-loop.svg
```

The plots in `docs/figures/` are generated from the results: `python -m rubricrl.analysis.figures all`.
