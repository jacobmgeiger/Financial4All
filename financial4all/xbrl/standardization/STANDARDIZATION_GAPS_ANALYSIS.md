# Standardization Gaps Analysis

## Overview
This document analyzes standardization features present in `edgartools` that we may be missing or could enhance in our system.

## Current Standardization Features

### ✅ What We Have
1. **StandardConcept Enum**: Canonical concept names with display labels
2. **Reverse Index**: O(1) lookups from XBRL tags to standard concepts
3. **Section Membership**: Context-aware disambiguation using statement sections
4. **Exclusions**: Tags marked as "DropThisItem" excluded from standardization
5. **Unmapped Tag Logging**: System for logging unmapped/ambiguous tags
6. **Company-Specific Mappings**: Priority-based company mappings (MSFT, TSLA, BRKA, AAPL)
7. **Hierarchy Rules**: Parent-child relationships in company mappings
8. **GAAP Mappings**: Comprehensive mappings from SynonymGroups (62 concepts, 212 tags)
9. **Label Similarity Matching**: SequenceMatcher-based similarity for inference
10. **Context-Aware Disambiguation**: Uses statement type and section for ambiguous tags

## Potential Gaps & Enhancements

### 1. SRT Taxonomy Support ⚠️ PARTIAL

**Status**: We mention SRT in namespace stripping but don't have dedicated handling.

**What edgartools has**:
- Dedicated SRT taxonomy data directory (`data/xbrl/srt/`)
- SRT taxonomy mappings for standardized reporting concepts
- Support for SEC Standardized Taxonomy (SRT) tags

**What we need**:
- SRT-specific concept mappings
- Recognition that `srt_*` tags may need different handling than `us-gaap_*` tags
- SRT taxonomy file support (if needed for specific industries)

**Priority**: Medium
**Impact**: Some companies use SRT tags, especially in regulated industries

---

### 2. Calculation-Based Validation ✅ IMPLEMENTED

**Status**: ✅ **COMPLETED** - Calculation validation system implemented and integrated.

**What edgartools has**:
- Uses calculation relationships to validate mappings
- Example: If `Revenue - COGS = Gross Profit`, validates that mapped concepts satisfy this relationship
- Cross-checks mappings against calculation formulas

**Implementation**:
- Created `CalculationValidator` class in `calculation_validation.py`
- Integrated into `ConceptMapper.map_concept()` with automatic validation
- Standard validation rules for common financial relationships (Gross Profit, Operating Income, Net Income, etc.)
- Special validation for Interest Income vs Operating Income (catches AAPL-style misclassifications)
- Calculation linkbase integration for dynamic validation
- Non-blocking: Logs warnings but doesn't prevent processing

**Files Created**:
- `financial4all/xbrl/standardization/calculation_validation.py`
- `financial4all/xbrl/standardization/CALCULATION_VALIDATION_README.md`

**Priority**: High ✅ **COMPLETED**
**Impact**: Catches mapping errors like the AAPL Interest Income issue early

---

### 3. Presentation Tree-Based Disambiguation ❌ MISSING

**Status**: We parse presentation trees but don't use them for standardization.

**What edgartools has**:
- Uses presentation tree structure to disambiguate concepts
- Leverages parent-child relationships in presentation linkbase
- Uses tree position/order to infer concept meaning

**What we need**:
```python
def disambiguate_with_presentation_tree(
    concept: str,
    presentation_tree: PresentationTree,
    context: Dict[str, Any]
) -> Optional[str]:
    """
    Use presentation tree structure to disambiguate ambiguous concepts.
    
    Example:
        If concept appears as child of "Revenue" in presentation tree,
        it's likely a revenue component, not a standalone revenue.
    """
    pass
```

**Priority**: Medium
**Impact**: Better disambiguation for hierarchical concepts (e.g., Revenue components)

---

### 4. Taxonomy Hierarchy Awareness ❌ MISSING

**Status**: We don't leverage the actual taxonomy hierarchy.

**What edgartools has**:
- Traverses taxonomy hierarchy (e.g., `us-gaap_SalesRevenueNet` is subtype of `us-gaap_Revenue`)
- Uses taxonomy relationships to infer mappings
- Leverages definition linkbase relationships

**What we need**:
```python
def infer_from_taxonomy_hierarchy(
    concept: str,
    taxonomy_hierarchy: Dict[str, List[str]]
) -> Optional[str]:
    """
    Infer standard concept from taxonomy parent-child relationships.
    
    Example:
        If concept is child of "Revenue" in taxonomy,
        map to StandardConcept.REVENUE
    """
    pass
```

**Priority**: Medium
**Impact**: More accurate mappings using official taxonomy structure

---

### 5. Automated Learning System ❌ MISSING

**Status**: We don't have automated batch processing to learn mappings.

**What edgartools has**:
- Batch job that processes XBRL filings
- Extracts concepts and proposes mappings
- Confidence scoring and validation pipeline
- Progressive refinement over time

**What we need**:
```python
class MappingLearner:
    """
    Automated system to learn mappings from XBRL filings.
    
    Features:
    - Batch processing of filings
    - Label similarity matching
    - Contextual rule inference
    - Cross-company pattern detection
    - Confidence scoring
    - Pending mappings for manual review
    """
    def learn_from_filings(self, filings: List[XBRL]) -> Dict[str, float]:
        """Process filings and propose new mappings with confidence scores."""
        pass
    
    def validate_mappings(self, proposed_mappings: Dict[str, float]) -> Dict[str, bool]:
        """Validate proposed mappings against calculation relationships."""
        pass
```

**Priority**: Low (Nice to have)
**Impact**: Would help discover new mappings automatically over time

---

### 6. ML-Based Similarity ❌ MISSING

**Status**: We use simple string matching (SequenceMatcher).

**What edgartools has**:
- ML embeddings (e.g., BERT) for label similarity
- Trained models on financial statement datasets
- Better handling of synonyms and variations

**What we need**:
```python
class MLSimilarityMatcher:
    """
    ML-based similarity matching for concept labels.
    
    Uses embeddings to find semantically similar concepts.
    """
    def similarity_score(self, label1: str, label2: str) -> float:
        """Calculate semantic similarity using ML embeddings."""
        pass
```

**Priority**: Low (Future enhancement)
**Impact**: Better matching for complex synonyms and variations

---

### 7. Cross-Company Pattern Detection ❌ MISSING

**Status**: We don't analyze patterns across multiple companies.

**What edgartools has**:
- Identifies recurring concepts across companies
- Statistical pattern detection (e.g., 90% of companies use `us-gaap_Revenue` for top-line)
- Uses frequency to boost confidence

**What we need**:
```python
def detect_cross_company_patterns(
    filings: List[XBRL],
    concept: str
) -> Dict[str, float]:
    """
    Detect patterns across companies for a concept.
    
    Returns:
        Dict mapping standard concepts to frequency/confidence
    """
    pass
```

**Priority**: Low (Nice to have)
**Impact**: Better confidence scoring for mappings

---

### 8. Statement Position Awareness ⚠️ PARTIAL

**Status**: We use section membership but not exact position/order.

**What edgartools has**:
- Uses position in statement (e.g., top-line = Revenue)
- Leverages order attribute from presentation linkbase
- Position-based heuristics for mapping

**What we need**:
```python
def infer_from_position(
    concept: str,
    position: int,
    statement_type: str
) -> Optional[str]:
    """
    Infer standard concept from position in statement.
    
    Example:
        Position 0 in IncomeStatement → Revenue
        Position 1 in IncomeStatement → Cost of Revenue
    """
    pass
```

**Priority**: Low
**Impact**: Minor improvement in mapping accuracy

---

## Recommended Implementation Priority

### Phase 1: High Priority (Immediate Impact)
1. **Calculation-Based Validation** ⭐⭐⭐ ✅ **COMPLETED**
   - ✅ Catches mapping errors early
   - ✅ Leverages existing calculation parsing
   - ✅ High ROI for effort

### Phase 2: Medium Priority (Significant Improvement)
2. **SRT Taxonomy Support** ⭐⭐
   - Needed for some companies/industries
   - Relatively straightforward to add
3. **Presentation Tree-Based Disambiguation** ⭐⭐
   - Better handling of hierarchical concepts
   - Uses existing presentation tree parsing
4. **Taxonomy Hierarchy Awareness** ⭐⭐
   - More accurate mappings using official structure
   - Requires definition linkbase parsing

### Phase 3: Low Priority (Nice to Have)
5. **Automated Learning System** ⭐
   - Long-term value but complex
   - Can be added incrementally
6. **ML-Based Similarity** ⭐
   - Future enhancement
   - Requires ML infrastructure
7. **Cross-Company Pattern Detection** ⭐
   - Useful for confidence scoring
   - Can be added to learning system

---

## Implementation Notes

### Calculation-Based Validation
- Integrate with existing `CalculationEngine` class
- Add validation step after mapping
- Flag suspicious mappings for review

### SRT Taxonomy Support
- Add SRT-specific mappings to `gaap_mappings.json` or separate file
- Update namespace detection to handle SRT tags
- Test with companies known to use SRT

### Presentation Tree Disambiguation
- Extend `ConceptMapper.map_concept()` to accept presentation tree
- Use tree structure for context-aware disambiguation
- Integrate with existing section membership system

---

## References
- [edgartools standardization design](https://github.com/dgunning/edgartools/tree/main/edgar/xbrl/design)
- [edgartools enhanced standardization](https://github.com/dgunning/edgartools/blob/main/edgar/xbrl/design/enhanced-standardization-design.md)
- [edgartools learning system](https://github.com/dgunning/edgartools/blob/main/edgar/xbrl/design/standardization.md)
