## **Gettting Started**
1. python==3.8+
2. ```pip install -r requirements.txt```
3. ```python qt_label.py```

## **How to Use**
1.  **Open Directory**: Click to select the directory containing images to be labeled.
2. **Annotation**: Set the label to "positive" (**1**), "negative" (**2**), or "shadow" (**3**) by pressing the corresponding number key. Cursor color changes (Green for positive, yellow for negative, and another color for shadow).
3. **Undo and Redo**: Use ```Ctrl + Z``` for **Undo** and ```Ctrl + Y``` for **Redo** to manage label changes.
4. **Navigation**: Use "**A**" (back) or "**D**" (next) keys to navigate through images.
5.  **Saving**: REMEMBER to either click **Save Data** button or Press ```Command/Ctrl + S``` to save the labeled data for the current image. But don't worry if you forget to do so, go back and the annotations you've already done will be safe and sound.

## **FAQ**
- Where would the labeled patches go? 
    - Labeled data goes to "Positive", "Negative", and "Shadow" folders alongside the images. 
- Can I UNDO?
    - Yes, use "Ctrl + Z" for Undo and "Ctrl + Y" for Redo to manage label changes.

- Oops. The app crashed / I closed the app somehow.
    - No worries. All states are maintained in the stats.csv and will be restored the next time you launch the app. Try the **slider bar** to navigate to where you ceased.
- I'm not sure whether it is leakage or not.
    - Along with going forwards and backward to see if there is a spreading trend, refer to the binary image I put aside the main widget. Those are difference maps derived from the first stage of the model. 
