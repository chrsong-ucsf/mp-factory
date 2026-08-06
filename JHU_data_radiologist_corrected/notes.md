# BDMAP Radiologist Mask Reuse Notes

Original radiologist notes regarding segmentation mask reusability across multi-phase scans in the BDMAP dataset.

---

## Mask Reusability & Derivation Rules

1. **`BDMAP_00242114.seg.nrrd`**
   * **Original Note:** `114是在113的基础上稍微修改了一下，15和16都可以用13`
   * **Translation:** Scan `114` was slightly modified based on scan `113` (`BDMAP_00242113`). Both scans `115` (`BDMAP_00242115`) and `116` (`BDMAP_00242116`) can use scan `113`'s annotation mask.

2. **`BDMAP_00242131.seg.nrrd`**
   * **Original Note:** `31-33 都可以用这个annotation`
   * **Translation:** Scans `131`, `132`, and `133` (`BDMAP_00242131`–`BDMAP_00242133`) can all use this annotation mask (`BDMAP_00242131.seg.nrrd`).

3. **`BDMAP_00242136.seg.nrrd`**
   * **Original Note:** `35-36 可以用同一个`
   * **Translation:** Scans `135` (`BDMAP_00242135`) and `136` (`BDMAP_00242136`) can use the same annotation mask (`BDMAP_00242136.seg.nrrd`).