# Sprite Editor
This is a visual tool for editting images with `.npy` format. 

Remember: if you can't execute the script, make sure that your current directory is `\Jaxtari`! 

For example:
```
C:\Work\Jaxtari> & C:/Python310/python.exe c:/Work/Jaxtari/spriteEditor/spriteEditor.py
```

## File Menu

The File Menu is located on the toolbar on the top of the window.

* **Open**: Open a file in `.npy` format. The file is an image file in either *RGB* or *RGBA* format. All values must be integers in $[0,255)$. If the file is in *RGB* format, it will be automatically converted into *RGBA*.
* **Save (Shortcut: Ctrl+S):**  Save the selected pixels into a `.npy` file in *RGBA* format.

## Edit Menu

The Edit Menu is located on the toolbar on the top of the window.

* **Undo (Shortcut: Ctrl+Z)**: undo the last operation.
* **Redo (Shortcut: Ctrl+Y)**: redo the last operation.

The state queue is displayed on the right-bottom corner of the window, and you can scroll through it. Undone states are shown with gray text.

## Tools Toolbar

The Tools toolbar is located below the canvas.

* **Zoom In (Shortcut: Ctrl+MouseScroll+)**: Zoom in the canvas.
* **Zoom Out (Shortcut: Ctrl+MouseScroll-)**: Zoom out the canvas.
* **Pencil**: Draw pixels using the current color set in the palette.
* **Magic Wand**: Select a continuous region of pixels w/ the same color. When this tool is selected, *Selection Toolbar* is displayed.
* **Rectangular Selection**: Select a rectangle. When this tool is selected, *Selection Toolbar* is displayed.
* **Dropper**: Click on a pixel to set it as the current color.
* **Select w/ same color**: Select all pixels in the image with the same color.
* **Color Indicator**: Click on it to show a palette that allows you to choose the RGB of current color.
* **Alpha Indicator**: Alpha value *a=x*, where $x \in \{0, ..., 255\}$. Click on it to choose the alpha of current color.

## Selection Toolbar

When *Magic Wand* or *Rectangular Selection* tool is selected, the *Selection Toolbar* is displayed below the *Tool Toolbar*.

* **Selection Modes**: Similar to New, Add, Subtract, Intersect selection modes in Adobe Photoshop.
* **Fill Selected**: Fill the selected pixels with the current color.
* **Save as Preset**: Save the current selection as preset.
* **Load from Preset**: Load the selection from preset.

## Appendix: List of Shortcuts
* **Control+A**: Select All 
* **Control+D**: Deselect All
* **Control+S**: Save current selection
* **Control+Mousewheel**: Zoom
* **Control+Y**: Redo
* **Control+Z**: Undo
* **Delete**: Set selected pixels as transparent


