#!/bin/bash
# This script documents the remaining C API changes needed for CAGRA and Tiered Index
# We'll apply them manually to maintain formatting consistency

echo "Remaining C API changes needed:"
echo ""
echo "1. CAGRA (c/src/neighbors/cagra.cpp):"
echo "   - Add BITMAP filter support to _search template (BITSET already exists)"
echo "   - Pattern: Same as IVF_PQ BITMAP implementation"
echo ""
echo "2. Tiered Index (c/src/neighbors/tiered_index.cpp):"
echo "   - Add BITMAP and BITSET filter support to _search template"
echo "   - Pattern: Same as IVF_PQ implementation"
echo ""
echo "3. Tests needed:"
echo "   - c/tests/neighbors/ann_cagra_c.cu: Add BuildSearchBitmapFiltered test"
echo "   - c/tests/neighbors/ann_tiered_index_c.cu: Create new file with both filter tests"
echo "   - c/tests/CMakeLists.txt: Register TIERED_INDEX_C_TEST"
echo ""
echo "All changes should preserve existing formatting - only modify/add necessary code"
