# from geopy.geocoders import GoogleV3
import censusgeocode as cg
# from pygris.geocode import geocode
import hashlib
import pandas as pd
import numpy as np
import time
from unidecode import unidecode
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import streamlit as st

st.title("PII Tokenization Testing")

uploaded_file = st.file_uploader("Upload a CSV file with fake data", type="csv")

start = time.time()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, dtype={'ZipCode': str})
    st.write("### Preview of Uploaded Data")
    st.write(df.head())

    latlong_decimalpts = 7


    def geocode_census(row):
        try:
            # result = cg.address(street=row['Address'], city=row['City'], state=row['State'], zip=row['ZipCode'])
            result = cg.onelineaddress(row['FullAddress'])
            census_tract = result[0]['geographies']['Census Tracts'][0]
            coordinates = result[0]['coordinates']
            return pd.Series({
                'FULLTRACT': census_tract['GEOID'],
                'TRACT': census_tract['TRACT'],
                'COUNTY': census_tract['COUNTY'],
                'STATE': census_tract['STATE'],
                'Latitude': round(coordinates['y'], latlong_decimalpts),
                'Longitude': round(coordinates['x'], latlong_decimalpts),
            })
        except Exception as e:
            print(f"Error processing {row}: {str(e)}")
            return pd.Series([None] * 6, index=['Latitude', 'Longitude', 'FULLTRACT', 'TRACT', 'COUNTY', 'STATE'])


    # def geocode_latitude(g_locator, address):
    #     location = g_locator.geocode(address)
    #     if location!=None:
    #       return location.latitude
    #     else:
    #       return np.nan
    #
    # def geocode_longitude(g_locator, address):
    #     location = g_locator.geocode(address)
    #     if location!=None:
    #       return location.longitude
    #     else:
    #       return np.nan

    df['FirstName_clean'] = df['FirstName'].str.replace('-', '', regex=True).str.replace(' ', '').str.lower().apply(
        unidecode)

    df['LastName_clean'] = df['LastName'].str.replace('-', '', regex=True).str.replace(' ', '').str.lower().apply(
        unidecode)

    df['DOB'] = pd.to_datetime(df['DOB'], format='%m/%d/%Y')
    df['BirthYear'] = df['DOB'].dt.year.round(0)
    df['DOB'] = df['DOB'].astype(str)
    df['SSN_clean'] = df['SSN'].replace('-', '', regex=True).astype(str)
    df['SSN_9digit'] = df['SSN_clean'].str.match(r'^\d{9}$')
    df['Phone_clean'] = df['PhoneNumber'].replace('-', '', regex=True).astype(str)
    df['Phone_10digit'] = df['Phone_clean'].str.match(r'^\d{10}$')

    df['FullName'] = df['FirstName_clean'] + df['LastName_clean']
    df['FullAddress'] = df[['Address', 'City', 'State', 'ZipCode']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    # t['FullNameAddress'] = t['FirstName'] + t['LastName'] + t['Address']
    # t['Latitude'] = t['FullAddress'].apply(lambda x:geocode_latitude(geolocator,x)).round(latlong_decimalpts).astype(str)
    # t['Longitude'] = t['FullAddress'].apply(lambda x:geocode_longitude(geolocator,x)).round(latlong_decimalpts).astype(str)

    # test making this a much bigger df
    # df = pd.DataFrame(np.repeat(df.values, 1300, axis=0))
    # df = df.columns
    # df = pd.concat([df]*13*3).sort_index().reset_index(drop=True)

    # get Census tract using geocoded address for each row
    df = pd.concat([
        df,
        df.apply(geocode_census, axis=1)
    ], axis=1)

    # for Token1
    df['FullNameSSN'] = df['FullName'] + df['SSN_clean']
    # for Token2
    df['FullNameDOB'] = df['FullName'] + df['DOB']
    # for Token3
    df['FullNameSSNDOB'] = df['FullName'] + df['SSN_clean'] + df['DOB']
    # for Token4
    df['FullNameCoords'] = df['FullName'] + df['Latitude'].astype(str) + df['Longitude'].astype(str)
    # for Token5
    df['LastNameDOB'] = df['LastName_clean'] + df['DOB']
    # for Token6
    df['FirstNameDOB'] = df['FirstName_clean'] + df['DOB']
    # for Token7
    df['FullNamePhone'] = df['FullName'] + df['Phone_clean']
    # for Token8
    df['FullNameCoordsPhone'] = df['FullName'] + df['Latitude'].astype(str) + df['Longitude'].astype(str) + df[
        'Phone_clean']
    # for Token10
    df['SSNDOB'] = df['SSN_clean'] + df['DOB']
    # for Token11
    df['AddressDOB'] = df['Latitude'].astype(str) + df['Longitude'].astype(str) + df['DOB']


    # def get_census_tract(row):
    #     try:
    #         result = cg.coordinates(x=row['Longitude'], y=row['Latitude'])
    #         census_tract = result['Census Tracts'][0]
    #         return pd.Series({
    #             'FULLTRACT': census_tract['GEOID'],
    #             'TRACT': census_tract['TRACT'],
    #             'COUNTY': census_tract['COUNTY'],
    #             'STATE': census_tract['STATE']
    #         })
    #     except Exception as e:
    #         print(f"Error processing {row}: {str(e)}")
    #         return pd.Series([None]*4, index=['FULLTRACT','TRACT','COUNTY','STATE'])
    #
    # # get Census tract using lat and long for each row
    # t = pd.concat([
    #     t,
    #     t.apply(get_census_tract, axis=1)
    # ], axis=1)

    # Deterministic Tokenization Function
    def deterministic_tokenize(text):
        if "nan" in text or "NaT" in text or text is None or text == '':
            return "missing"
        return hashlib.sha256(text.encode()).hexdigest()


    # Apply Tokenization to PII Columns
    # First name, last name, SSN
    df['Token1'] = df['FullNameSSN'].apply(deterministic_tokenize)
    # First name, last name, DOB
    df['Token2'] = df['FullNameDOB'].apply(deterministic_tokenize)
    # First name, last name, SSN, DOB
    df['Token3'] = df['FullNameSSNDOB'].apply(deterministic_tokenize)
    # First name, last name, Lat, Long
    df['Token4'] = df['FullNameCoords'].apply(deterministic_tokenize)
    # Last name, DOB
    df['Token5'] = df['LastNameDOB'].apply(deterministic_tokenize)
    # First name, DOB
    df['Token6'] = df['FirstNameDOB'].apply(deterministic_tokenize)
    # First name, last name, phone number
    df['Token7'] = df['FullNamePhone'].apply(deterministic_tokenize)
    # First name, last name, Lat, Long, phone number
    df['Token8'] = df['FullNameCoordsPhone'].apply(deterministic_tokenize)
    # SSN
    df['Token9'] = df['SSN_clean'].apply(deterministic_tokenize)
    # SSN, DOB
    df['Token10'] = df['SSNDOB'].apply(deterministic_tokenize)
    # Lat, Long, DOB
    df['Token11'] = df['AddressDOB'].apply(deterministic_tokenize)
    # DOB
    df['Token12'] = df['DOB'].apply(deterministic_tokenize)

    # # Step 1: Identify duplicates in Token5
    # df['Token5duplicates'] = df.duplicated('Token5', keep=False)
    #
    # # Step 2: Fuzzy matching function
    # def get_fuzzy_similarity(value, unique_values):
    #     # Find the best match and its similarity score
    #     best_match = process.extractOne(value, unique_values)
    #     if best_match:
    #         return best_match[1]  # Return the similarity score
    #     return 0  # Return 0 if no match is found (edge case)
    #
    # # Step 3: Apply fuzzy matching only for rows with duplicates in `other_column`
    # df['fuzzy_similarity'] = df.apply(
    #     lambda row: get_fuzzy_similarity(row['FullName'], df[df['Token5duplicates']]['Token5'].unique())
    #     if row['Token5duplicates'] else 0, axis=1
    # )

    # Token 5 (last name + DOB) is same -- fuzzy matching starts here
    Token5_nomissing = df[df['Token5'] != 'missing']
    Token5_duplicate_cats = Token5_nomissing['Token5'][Token5_nomissing.duplicated('Token5', keep=False)]

    df['Token5_NameSimilarity'] = None  # Initialize new column
    Token5_duplicates = df[df['Token5'].isin(Token5_duplicate_cats)]

    # for category, group in Token5_duplicates.groupby('Token5'):
    #     names = group['FullName'].tolist()
    #     matches = {desc: process.extractOne(desc, names) for desc in names}
    #     df.loc[df['Token5'] == category, 'FullNameSimilarity'] = df['FullName'].map(lambda x: matches[x][1])

    for category, group in df[df['Token5'].isin(Token5_duplicate_cats)].groupby('Token5'):
        descriptions = group['FullName'].tolist()
        ratios = {}

        for desc in descriptions:
            matches = process.extract(desc, descriptions, limit=2)  # Get top 2 matches (including itself)
            best_match = matches[1] if len(matches) > 1 else (desc, 0)  # Pick the best match that isn't itself
            ratios[desc] = best_match[1]  # Store only the highest similarity score

        for index, row in group.iterrows():
            # best_match, score = matches[row['FullName']] if matches[row['FullName']] else ("", 0)
            # df.at[index, 'Fuzzy_Match'] = best_match
            df.at[index, 'Token5_NameSimilarity'] = ratios[row['FullName']]

    # Token 12 (DOB) is same -- fuzzy matching starts here
    Token12_nomissing = df[df['Token12'] != 'missing']
    Token12_duplicate_cats = Token12_nomissing['Token12'][Token12_nomissing.duplicated('Token12', keep=False)]

    df['Token12_NameSimilarity'] = None  # Initialize new column
    Token5_duplicates = df[df['Token12'].isin(Token12_duplicate_cats)]

    # for category, group in Token5_duplicates.groupby('Token12'):
    #     names = group['FullName'].tolist()
    #     matches = {desc: process.extractOne(desc, names) for desc in names}
    #     df.loc[df['Token12'] == category, 'FullNameSimilarity'] = df['FullName'].map(lambda x: matches[x][1])

    for category, group in df[df['Token12'].isin(Token12_duplicate_cats)].groupby('Token12'):
        descriptions = group['FullName'].tolist()
        ratios = {}

        for desc in descriptions:
            matches = process.extract(desc, descriptions, limit=2)  # Get top 2 matches (including itself)
            best_match = matches[1] if len(matches) > 1 else (desc, 0)  # Pick the best match that isn't itself
            ratios[desc] = best_match[1]  # Store only the highest similarity score

        for index, row in group.iterrows():
            # best_match, score = matches[row['FullName']] if matches[row['FullName']] else ("", 0)
            # df.at[index, 'Fuzzy_Match'] = best_match
            df.at[index, 'Token12_NameSimilarity'] = ratios[row['FullName']]

    # def name_similarity(group):
    #     if len(group) > 1:
    #         scores = []
    #         for i in range(len(group)):
    #             for j in range(i + 1, len(group)):
    #                 score = fuzz.ratio(group.iloc[i], group.iloc[j])
    #                 scores.append((group.iloc[i], group.iloc[j], score))
    #         return scores
    #     return []
    #
    # def test_function(group):
    #     if len(group) > 1:
    #         for i in range(len(group)):
    #             for j in range(i + 1, len(group)):
    #                 return fuzz.ratio(group.iloc[i], group.iloc[j])
    #
    # # Token5_duplicate_groups = (
    #
    # Token5_duplicates.groupby('FullName').apply(name_similarity)
    #
    # Token5_similarity_results = []
    # for group in Token5_duplicate_groups:
    #     for text1, text2, score in group:
    #         Token5_similarity_results.append({'text1': text1, 'text2': text2, 'similarity_score': score})
    #
    # Token5_sim_df = pd.DataFrame(Token5_similarity_results)
    #
    # df = df.merge(Token5_sim_df, left_on='Token5', right_on='text1', how='left')

    cols_for_deid = ['Org', 'OrgID'] + [col for col in df.columns if col.startswith('Token')] + ['BirthYear', 'Gender',
                                                                                                 'FULLTRACT', 'TRACT',
                                                                                         'COUNTY', 'STATE',
                                                                                                 'ZipCode']

    deidentified_df = df[cols_for_deid]

    st.write("### Processed Deidentified Data - click to download")
    st.write(deidentified_df.head())

    # df.to_csv('TestOutputFull_v3.csv', index=False)
    # deidentified_df.to_csv('TestDeidentified_v3.csv', index=False)

    end = time.time()

    print(f"Execution time: {end - start:.6f} seconds")
    # when running 8 rows x 13 = 104: 39 seconds
    # when running 8 rows x 13 x 3 = 312: 115 seconds

