import pytest
from drug_molecule_gen.config import config
from drug_molecule_gen.preprocessing.data_management import load_dataset
import drug_molecule_gen.preprocessing.preprocessor as pp
from drug_molecule_gen.predict import generate_molecule
import random

@pytest.fixture
def single_molecule_generation():
    cv_data=load_dataset(config.TEST_FILE)
    preprocess = pp.preprocess_data()
    preprocess.fit()
    X_cv, Y_cv = preprocess.transform(cv_data["X"],cv_data["Y"])
    single_input = X_cv[random.randint(0,X_cv.shape[0]),:]
    single_input= single_input.reshape(1,single_input.shape[0])

    gen_mol = generate_molecule(single_input,1)

    return gen_mol

def test_is_molecule_generation_none(single_molecule_generation):
    assert single_molecule_generation is not None


def test_gen_molcule_dtype(single_molecule_generation):
    assert  not isinstance(single_molecule_generation.get("Generated Valid Molecules"),str)
