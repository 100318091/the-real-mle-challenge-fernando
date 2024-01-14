import pandas
from src.training.fitting import room_type_and_neigh_mapping


class TestRoomTypeAndNeighMapping:
    # Given a DataFrame with a 'neighbourhood' column and a 'room_type' column, the function should map the values of the 'neighbourhood' column to the corresponding values in the 'MAP_NEIGHB' dictionary, and map the values of the 'room_type' column to the corresponding values in the 'MAP_ROOM_TYPE' dictionary.
    def test_mapping_values(self):
        # Arrange
        df = pandas.DataFrame(
            {
                "neighbourhood": ["Bronx", "Queens", "Manhattan"],
                "room_type": ["Entire home/apt", "Private room", "Shared room"],
            }
        )
        expected_df = pandas.DataFrame(
            {"neighbourhood": [1, 2, 5], "room_type": [3, 2, 1]}
        )

        # Act
        result_df = room_type_and_neigh_mapping(df)

        # Assert
        pandas.testing.assert_frame_equal(result_df, expected_df)

    # TODO: Add more tests.
