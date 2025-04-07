import requests
import pandas as pd
from tqdm import tqdm
import time
def main():

    API_KEY = '**************************'
    BASE_URL = 'https://rebrickable.com/api/v3/'
    headers = {'Authorization': f'key {API_KEY}'}

    # Fetch LEGO set inventory
    def fetch_set_inventory(set_num):
        url = f'{BASE_URL}lego/sets/{set_num}/parts/?page_size=1000'
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    # Fetch LEGO set details
    def fetch_set_details(set_num):
        url = f'{BASE_URL}lego/sets/{set_num}/'
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    # Get a large list of LEGO sets (pagination)
    def fetch_all_sets(max_sets=10000):
        sets = []
        page = 1
        page_size = 1000
        while len(sets) < max_sets:
            url = f'{BASE_URL}lego/sets/?page={page}&page_size={page_size}'
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            sets.extend(data['results'])
            if not data['next']:
                break
            page += 1
            time.sleep(1)  # Rate limit respect
        return sets[:max_sets]

    all_sets = fetch_all_sets(1000)
    all_data = []

    for lego_set in tqdm(all_sets, desc='Fetching LEGO sets'):
        set_num = lego_set['set_num']
        try:
            inventory = fetch_set_inventory(set_num)
            for piece in inventory['results']:
                piece_data = {
                    'set_number': set_num,
                    'set_name': lego_set['name'],
                    'total_pieces_in_set': lego_set['num_parts'],
                    'piece_part_num': piece['part']['part_num'],
                    'piece_name': piece['part']['name'],
                    'piece_color': piece['color']['name'],
                    'quantity_of_piece': piece['quantity']
                }
                all_data.append(piece_data)
            time.sleep(0.5)  # Adjust delay as needed
        except requests.HTTPError as e:
            print(f'Failed to fetch set {set_num}: {e}')

    # Save DataFrame as CSV
    df = pd.DataFrame(all_data)
    df.to_csv('lego_sets_pieces.csv', index=False)
    print('File lego_sets_pieces.csv created successfully!')

if __name__ == "__main__":
    main()