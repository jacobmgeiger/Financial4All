import xml.etree.ElementTree as ET
import json
import sys
from collections import defaultdict

# xblr_prep/xblr_main.py
# This script is the first step in the data enrichment pipeline. Its purpose is to
# parse the entire US-GAAP XBRL taxonomy to find all defined calculation relationships
# (e.g., Gross Profit = Revenues - Cost of Revenue). It processes dozens of `cal-2025.xml`
# files, each containing formulas for a specific financial statement or disclosure.
#
# The script intelligently extracts these formulas, groups them by the "parent" metric,
# and ensures that duplicate formulas (which can appear in different contexts) are not
# added. The final output is a single, comprehensive JSON file that serves as the
# "brain" for the solver in the main data_loader.


def parse_calculation_linkbase(file_path):
    """
    Parses a single XBRL calculation linkbase XML file to extract all distinct
    calculation relationships, grouped by their XBRL role (i.e., the specific
    financial statement or disclosure they belong to).

    Args:
        file_path (str): The path to the calculation linkbase XML file.

    Returns:
        dict: A dictionary where keys are 'parent' elements (e.g., 'GrossProfit') and
              values are lists of possible formulas. Each formula is a dictionary
              containing its role and its constituent children.
    """
    try:
        tree = ET.parse(file_path)
    except ET.ParseError as e:
        print(f"Error parsing XML file: {file_path}", file=sys.stderr)
        print(e, file=sys.stderr)
        return {}

    root = tree.getroot()

    namespaces = {
        "link": "http://www.xbrl.org/2003/linkbase",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    # Structure: { parent_metric: [ { "role": role_uri, "children": [...] } ] }
    calculations = defaultdict(list)

    # A calculationLink represents a set of relationships for a specific statement (role).
    for calc_link in root.findall("link:calculationLink", namespaces):
        role_uri = calc_link.get("{" + namespaces["xlink"] + "}role")
        locators = {}

        # Step 1: Build a map of xlink:label to the actual metric name for this role.
        # The XBRL file uses internal labels, which we need to resolve to human-readable
        # and machine-readable metric names (e.g., 'us-gaap_Revenues').
        for loc in calc_link.findall("link:loc", namespaces):
            label = loc.get("{" + namespaces["xlink"] + "}label")
            href = loc.get("{" + namespaces["xlink"] + "}href")
            # Ensure href is not None before processing
            if href:
                element_name = href.split("#")[-1].replace("us-gaap_", "")
                locators[label] = element_name

        # Step 2: Group all calculation arcs by their parent (the 'from' attribute)
        # within this specific role. Each arc represents a parent-child relationship.
        formulas_in_role = defaultdict(list)
        for calc_arc in calc_link.findall("link:calculationArc", namespaces):
            parent_label = calc_arc.get("{" + namespaces["xlink"] + "}from")
            child_label = calc_arc.get("{" + namespaces["xlink"] + "}to")

            parent_element = locators.get(parent_label)
            child_element = locators.get(child_label)

            if parent_element and child_element:
                formulas_in_role[parent_element].append(
                    {
                        "child": child_element,
                        "weight": float(calc_arc.get("weight", 1.0)),
                    }
                )

        # Step 3: Add the discovered formulas to the main dictionary, ensuring that
        # we do not add duplicate formulas for the same parent.
        for parent, children in formulas_in_role.items():
            # Sort children for consistent comparison to reliably detect duplicates.
            new_formula = {
                "role": role_uri,
                "children": sorted(children, key=lambda x: x["child"]),
            }

            # To efficiently check for duplicates, we compare the set of child names.
            new_children_set = frozenset(c["child"] for c in new_formula["children"])
            is_duplicate = any(
                new_children_set
                == frozenset(c["child"] for c in existing_formula["children"])
                for existing_formula in calculations[parent]
            )

            if not is_duplicate:
                calculations[parent].append(new_formula)

    return calculations


if __name__ == "__main__":
    # This list contains all the calculation linkbase files from the 2025 US-GAAP XBRL taxonomy.
    # Processing all of them is crucial to building a complete formula set.
    cal_files_to_process = [
        "xblr_prep/us-gaap-2025/ebp/stm/us-gaap-ebp-stm-scnaab-cal-2025.xml",
        "xblr_prep/us-gaap-2025/ebp/stm/us-gaap-ebp-stm-snaab-cal-2025.xml",
        "xblr_prep/us-gaap-2025/ebp/dis/us-gaap-ebp-dis-debt-cal-2025.xml",
        "xblr_prep/us-gaap-2025/ebp/dis/us-gaap-ebp-dis-fvnav-cal-2025.xml",
        "xblr_prep/us-gaap-2025/ebp/dis/us-gaap-ebp-dis-derivative-cal-2025.xml",
        "xblr_prep/us-gaap-2025/ebp/dis/us-gaap-ebp-dis-recform5500-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-sfp-dbo-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-soi1-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-sfp-dbo1-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-soi-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-spc-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-soc-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-scf-inv-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-sfp-cls1-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-sfp-ibo-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-soc3-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-soc4-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-scf-dbo-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-sfp-cls-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-scf-indir-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-sfp-cls2-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-soc5-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-soc2-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-sheci-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-sheci2-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-scf-dir-cal-2025.xml",
        "xblr_prep/us-gaap-2025/stm/us-gaap-stm-soi4-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-rlnro-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-fs-bd-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-bsoff1-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-othliab5-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-ts-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-inv-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-inctax-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-cc2-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-fs-interest-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-ocpfs-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-sr-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-cecl-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-ides-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-invco-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-re-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-ni-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-ppe-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-cc-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-ides1-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-hco-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-ceclcalc2-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-fs-ins-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-fs-fhlb-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-fs-bt-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-debt-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-bc-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-equity-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-leas-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-bsoff-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-fifvd-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-fs-bt1-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-oi-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-fs-bd3-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-ceclcalc3l-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-diha-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-ei-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-pc-supp-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-schedoi-hold-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-regop-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-foct-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-crcsbp-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-crcrb-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-rcc-cal-2025.xml",
        "xblr_prep/us-gaap-2025/dis/us-gaap-dis-dr-cal-2025.xml",
    ]

    final_calcs = defaultdict(list)

    for f_path in cal_files_to_process:
        print(f"Processing {f_path}...", file=sys.stderr)
        calcs = parse_calculation_linkbase(f_path)
        for parent, formulas in calcs.items():
            # Merge formulas from the current file into the final dictionary,
            # ensuring we do not add duplicate formulas across different files.
            for new_formula in formulas:
                new_children_set = frozenset(
                    c["child"] for c in new_formula["children"]
                )
                is_duplicate = any(
                    new_children_set
                    == frozenset(c["child"] for c in existing_formula["children"])
                    for existing_formula in final_calcs[parent]
                )
                if not is_duplicate:
                    final_calcs[parent].append(new_formula)

    # Output the final, comprehensive dictionary of formulas as a JSON object to standard output.
    # This can then be redirected to a file (e.g., income_statement_formulas.json).
    print(json.dumps(final_calcs, indent=4))
