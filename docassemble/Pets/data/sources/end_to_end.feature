@end_to_end
Feature: Create a pet characteristics document

  # Seeded with ALDashboard's static fixture generator, then reviewed against the interview.
  Scenario: Complete the interview and reach its final document screen
    Given I start the interview at "multiple_row_retrieval.yml"
    And the maximum seconds for each Step in this Scenario is 240
    And the user gets to "download" with this data:
      | var | value |
      | pet.new_item_name | Milo |
      | pet['Milo'].species | Cat |
      | pet['Milo'].breed | Siamese |
      | pet['Milo'].label_ids['1'] | True |
      | pet.there_is_another | False |
    Then I should see the phrase "Your form is ready to download."
