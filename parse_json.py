
import json

json_string = '''{
  "plan": [
    {
      "placement_name": "First Screen Masthead - US",
      "audience_segment": "WBHE - The Flash - Superhero/Action PVOD/EST Viewers OR Superhero/DC/Marvel Fans (USID: 245802)",
      "start_datetime": "2025-06-15",
      "end_datetime": "2025-06-27",
      "rate": 24,
      "budget_allocated": 25350,
      "impressions": 1034701,
      "max_impression": 1034701,
      "additional_notes": "High impact primary placement targeting superhero/action fans for full campaign duration"
    },
    {
      "placement_name": "Universal Guide Masthead - US",
      "audience_segment": "WBHE - The Flash - Superhero/Action PVOD/EST Viewers OR Superhero/DC/Marvel Fans (USID: 245802)",
      "start_datetime": "2025-06-15",
      "end_datetime": "2025-06-27",
      "rate": 22,
      "budget_allocated": 58381,
      "impressions": 2653698,
      "max_impression": 2653698,
      "additional_notes": "Secondary high impact placement providing additional reach throughout campaign period"
    },
    {
      "placement_name": "Universal Guide Masthead - US (Weekend Heavy-Up)",
      "audience_segment": "WBHE - The Flash - Superhero/Action PVOD/EST Viewers OR Superhero/DC/Marvel Fans (USID: 245802)",
      "start_datetime": "2025-06-20",
      "end_datetime": "2025-06-21",
      "rate": 22,
      "budget_allocated": 4947,
      "impressions": 224855,
      "max_impression": 224855,
      "additional_notes": "Weekend heavy-up as specified in RFP for increased visibility during peak conversion period"
    },
    {
      "placement_name": "Apps Store Masthead - US",
      "audience_segment": "WBHE - The Flash - Superhero/Action PVOD/EST Viewers OR Superhero/DC/Marvel Fans (USID: 245802)",
      "start_datetime": "2025-06-15",
      "end_datetime": "2025-06-27",
      "rate": 24,
      "budget_allocated": 3973,
      "impressions": 162152,
      "max_impression": 162152,
      "additional_notes": "Complementary high-impact placement to extend reach across different Samsung TV touchpoints"
    }
  ]
}'''
data = json.loads(json_string)

placement_names = []
for item in data['plan']:
    placement_names.append(item['placement_name'])

print(placement_names)