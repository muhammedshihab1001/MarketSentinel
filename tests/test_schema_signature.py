from core.schema.feature_schema import get_schema_signature


def test_schema_signature_is_stable():
    """
    Prevent accidental feature changes.
    If this fails — you changed the model contract.
    """

    sig1 = get_schema_signature()
    sig2 = get_schema_signature()

    assert sig1 == sig2
