import pytest
from unittest.mock import patch
from gigabind.main import Gigabind


@patch.object(Gigabind, "model", create=True)
def test_run_text(mock_model):
    mock_model.return_value = {"embeddings": "test"}
    gigabind = Gigabind()
    response = gigabind.run(text="Hello, world!")
    mock_model.assert_called_once_with({"text": "Hello, world!"})
    assert response == {
        "embeddings": "test",
        "modality_type": ["text"],
        "model_name": "gigabind-huge",
    }


@patch.object(Gigabind, "model", create=True)
def test_run_img(mock_model):
    mock_model.return_value = {"embeddings": "test"}
    gigabind = Gigabind()
    response = gigabind.run(img=".assets/bird_image.jpg")
    mock_model.assert_called_once_with({"img": ".assets/bird_image.jpg"})
    assert response == {
        "embeddings": "test",
        "modality_type": ["img"],
        "model_name": "gigabind-huge",
    }


@patch.object(Gigabind, "model", create=True)
def test_run_audio(mock_model):
    mock_model.return_value = {"embeddings": "test"}
    gigabind = Gigabind()
    response = gigabind.run(audio=".assets/bird_audio.wav")
    mock_model.assert_called_once_with({"audio": ".assets/bird_audio.wav"})
    assert response == {
        "embeddings": "test",
        "modality_type": ["audio"],
        "model_name": "gigabind-huge",
    }


@patch.object(Gigabind, "model", create=True)
def test_run_all(mock_model):
    mock_model.return_value = {"embeddings": "test"}
    gigabind = Gigabind()
    response = gigabind.run(
        text="Hello, world!",
        img=".assets/bird_image.jpg",
        audio=".assets/bird_audio.wav",
    )
    mock_model.assert_called_once_with(
        {
            "text": "Hello, world!",
            "img": ".assets/bird_image.jpg",
            "audio": ".assets/bird_audio.wav",
        }
    )
    assert response == {
        "embeddings": "test",
        "modality_type": ["text", "img", "audio"],
        "model_name": "gigabind-huge",
    }


@patch.object(Gigabind, "model", create=True)
def test_run_exception(mock_model):
    mock_model.side_effect = Exception("Test exception")
    gigabind = Gigabind()
    with pytest.raises(Exception) as e:
        gigabind.run(text="Hello, world!")
    assert str(e.value) == "Test exception"


@patch.object(Gigabind, "model", create=True)
def test_run_empty_input(mock_model):
    mock_model.return_value = {"embeddings": "test"}
    gigabind = Gigabind()
    response = gigabind.run()
    mock_model.assert_not_called()
    assert response == {
        "embeddings": None,
        "modality_type": [],
        "model_name": "gigabind-huge",
    }


@patch.object(Gigabind, "model", create=True)
def test_run_invalid_modality(mock_model):
    mock_model.return_value = {"embeddings": "test"}
    gigabind = Gigabind()
    with pytest.raises(ValueError) as e:
        gigabind.run(invalid="Invalid input")
    assert str(e.value) == "Invalid modality: invalid"


@patch.object(Gigabind, "model", create=True)
def test_run_multiple_modalities(mock_model):
    mock_model.return_value = {"embeddings": "test"}
    gigabind = Gigabind()
    response = gigabind.run(text="Hello, world!", img=".assets/car_image.jpg")
    mock_model.assert_called_once_with(
        {"text": "Hello, world!", "img": ".assets/car_image.jpg"}
    )
    assert response == {
        "embeddings": "test",
        "modality_type": ["text", "img"],
        "model_name": "gigabind-huge",
    }
