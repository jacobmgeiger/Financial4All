# financial4all/xbrl/standardization/__init__.py
"""
Comprehensive standardization strategies for XBRL concepts.

This package provides advanced standardization capabilities matching edgartools'
comprehensive approach, including:
- StandardConcept enum with display labels
- Reverse index with ambiguous detection
- Section membership for context-aware disambiguation
- Exclusions for tags that shouldn't be standardized
- Unmapped tag logging for continuous improvement
- Company-specific mappings with priority scoring

Note: This package extends the parent standardization.py module. The parent module
contains SynonymGroups and related functionality, while this package adds advanced
features like reverse index, section membership, and unmapped logging.
"""

# Import new advanced features
from .standard_concepts import StandardConcept
from .exclusions import EXCLUDED_TAGS, should_exclude
from .sections import (
    SectionMembership, get_section_membership,
    get_section_for_concept, get_statement_for_concept,
    is_current, is_asset
)
from .reverse_index import (
    ReverseIndex, MappingResult, get_reverse_index,
    lookup, get_standard_concept, get_display_name
)
from .unmapped_logger import (
    UnmappedTagLogger, UnmappedTagEntry, AmbiguousResolutionEntry,
    get_unmapped_logger, log_unmapped, log_ambiguous
)
from .core import MappingStore, ConceptMapper

# Re-export parent module functions for backward compatibility
# Import from parent standardization.py module directly using importlib
try:
    import importlib.util
    import os
    import sys
    
    # Get the parent module file path
    parent_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'standardization.py'))
    
    # Create a unique module name to avoid conflicts
    parent_module_name = 'financial4all.xbrl.standardization_module'
    
    # Check if already imported
    if parent_module_name not in sys.modules:
        # Load the parent module file
        spec = importlib.util.spec_from_file_location(parent_module_name, parent_file)
        if spec and spec.loader:
            # Add parent directory to path for relative imports
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            original_path = sys.path[:]
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            try:
                parent_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(parent_module)
                sys.modules[parent_module_name] = parent_module
                
                # Export key functions and classes
                SynonymGroup = parent_module.SynonymGroup
                SynonymGroups = parent_module.SynonymGroups
                ConceptInfo = parent_module.ConceptInfo
                StandardizationStore = parent_module.StandardizationStore
                get_synonym_groups = parent_module.get_synonym_groups
                get_default_store = parent_module.get_default_store
            finally:
                # Restore original path
                sys.path[:] = original_path
        else:
            raise ImportError("Could not load parent standardization module")
    else:
        # Already imported, use cached version
        parent_module = sys.modules[parent_module_name]
        SynonymGroup = parent_module.SynonymGroup
        SynonymGroups = parent_module.SynonymGroups
        ConceptInfo = parent_module.ConceptInfo
        StandardizationStore = parent_module.StandardizationStore
        get_synonym_groups = parent_module.get_synonym_groups
        get_default_store = parent_module.get_default_store
except Exception as e:
    # If import fails, try alternative: import via module name after ensuring path
    try:
        import sys
        import os
        # Ensure parent directory is in path
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Import using the module name (this will import the .py file, not the package)
        # We need to temporarily rename/remove the package from the path
        import importlib
        import importlib.machinery
        
        # Force import of the .py file
        loader = importlib.machinery.SourceFileLoader(
            'financial4all.xbrl.standardization_py',
            os.path.join(parent_dir, 'standardization.py')
        )
        parent_module = loader.load_module('financial4all.xbrl.standardization_py')
        
        SynonymGroup = parent_module.SynonymGroup
        SynonymGroups = parent_module.SynonymGroups
        ConceptInfo = parent_module.ConceptInfo
        StandardizationStore = parent_module.StandardizationStore
        get_synonym_groups = parent_module.get_synonym_groups
        get_default_store = parent_module.get_default_store
    except Exception:
        # Last resort: raise a clear error
        raise ImportError(
            "Could not import parent standardization module. "
            "This is likely due to a package/module naming conflict. "
            "Please import directly from 'financial4all.xbrl.standardization' module."
        ) from e

__all__ = [
    # Standard concepts
    "StandardConcept",
    
    # Exclusions
    "EXCLUDED_TAGS",
    "should_exclude",
    
    # Sections
    "SectionMembership",
    "get_section_membership",
    "get_section_for_concept",
    "get_statement_for_concept",
    "is_current",
    "is_asset",
    
    # Reverse index
    "ReverseIndex",
    "MappingResult",
    "get_reverse_index",
    "lookup",
    "get_standard_concept",
    "get_display_name",
    
    # Unmapped logger
    "UnmappedTagLogger",
    "UnmappedTagEntry",
    "AmbiguousResolutionEntry",
    "get_unmapped_logger",
    "log_unmapped",
    "log_ambiguous",
    
    # Core
    "MappingStore",
    "ConceptMapper",
    
    # Parent module exports (if available)
    "SynonymGroup",
    "SynonymGroups",
    "ConceptInfo",
    "StandardizationStore",
    "get_synonym_groups",
    "get_default_store",
]
