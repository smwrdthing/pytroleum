import os
from typing import Any
from CoolProp.CoolProp import AbstractState


def generator_CoolStub(force: bool = False) -> None:
    """CoolStub.pyi generator

    Function to generate dummy CoolStub file. Instance of AbstractState class from
    CoolProp is an object of cython as well as PyPhaseEnvelopeData. No conventional and
    robust tools for stubs generation from cython source were found, this is why this
    function was written.

    DISCLAIMER :
    This generator basically inspects dirs of objects and creates text file with .pyi
    extension containing class descriptions for classes of interests. Since it is
    quite hard to make correct stub this function aims to generate "dummy" stub with all
    types being defaulted to "Any" and all methods signatures defaulted to
    (self, *args, **kwargs). User should adjust type annotations manually in generated
    file to represent expected inputs correctly.

    'CoolStub' name is hardcoded instead of accepting file name as input variable to
    prevent user from creation of file with erroneous name. If different name requiered
    consider renaming generated file or changing hardcoded name in the function.
    Destination directory is forced to be directory of the file containing this function
    for same reason.

    Parameters
    ----------
    force: bool = False
        Variable to determine behavior of function. By default function checks if stub
        file with hardcoded name already exists. If it does - execution is aborted,
        otherwise new stub is generated and saved. If force is True check does not happen,
        so if stub is already here it will be overriden.
    """

    # Harcoded default name for stub
    fname = 'CoolStub.pyi'
    # Get path where script resides
    script_path = os.path.dirname(os.path.realpath(__file__))
    # Construct path for file
    fpath = script_path + '\\' + fname

    # Check if file exist in case force option is False, abandon function if it does
    if not force:
        if 'CoolStub.pyi' in os.listdir(script_path):
            print("CoolStub generator is abandoned, no new files were written")
            return

    # We create an entry to objects, so we can access their dir. Passed arguments do not
    # really matter as we are interested in metadata of objects
    AS = AbstractState('HEOS', 'Methane&Ethane')
    AS.set_mole_fractions([0.6, 0.4])

    # We want to create class in the stub for phase envelope too, as we want to make
    # a stub class for it as well
    AS.build_phase_envelope("")
    PE = AS.get_phase_envelope_data()

    # Reading respective dirs
    contentAS = dir(AS)
    contentPE = dir(PE)

    # Now we create entries for stub-files texts, both start in the same fashion
    importing_code = 'from __future__ import annotations\n' + \
        'from typing import Any\n'  # for annotatinos
    stubcodeAS = 'class AbstractState:\n'
    stubcodePE = 'class PyPhaseEnvelopeData:'

    # Some convenience variables
    space = " "
    indent = 4*space
    default_args = "(self, *args: Any, **kwargs: Any) -> Any:"
    ellipsis = '...'

    # Genration of stub file content for AbstractState class
    # Do __init__ outside of loop
    new_stub_method_code = (
        '\n' + indent + 'def' + space + '__init__' +
        '(self, backend: str, fluid: str) -> None:' +
        '\n' + 2*indent + ellipsis + '\n'
    )
    stubcodeAS += new_stub_method_code
    for method_name in contentAS:
        if '__' not in method_name:  # throw other dunders out of the window
            new_stub_method_code = (
                '\n' + indent + 'def' + space + method_name +
                default_args + '\n' + 2*indent + ellipsis + '\n'
            )
            stubcodeAS += new_stub_method_code
    stubcodeAS += '\n'

    # Things are easier with PyPhaseEnvelopeData, we do not deal with methods here
    # and all attributes have type list[float]
    for attr_name in contentPE:
        if '__' not in attr_name:  # throw dunders out of the window
            new_attr_name = '\n'+indent+attr_name + ': list[float]'
            stubcodePE += new_attr_name
    stubcodePE += '\n'

    stubcode = importing_code + '\n\n' + stubcodeAS + '\n' + stubcodePE

    # Writing stub
    print("Writing new stub for CoolProp...")
    with open(fpath, 'w', encoding='utf-8') as f:
        f.writelines(stubcode)
