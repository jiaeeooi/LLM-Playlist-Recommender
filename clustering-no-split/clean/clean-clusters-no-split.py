#########################################
# Code to remove miscellaneous clusters #
#########################################

import os
import csv

def clean_clusters(input_file, output_file, threshold):
    with open(input_file, 'r', newline='', encoding='utf8') as infile, \
         open(output_file, 'w', newline='', encoding='utf8') as outfile:

        # The part with the csv readers/writers has been adapted from chatGPT
        reader = csv.DictReader(infile, delimiter=',')
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()

        #Filter the lines based on the threshold
        for row in reader:
            try:
                #convert the percentages in floats to avoid errors
                percentage = float(row["Exact Match Percentage"].replace('%', '').strip())
                if percentage > threshold:#filter
                    writer.writerow(row)
            except ValueError:
                continue

def main():
    #clusters_dir = "/home/vellard/playlist_continuation/clustering-no-split/analysis/200/"
    #output_dir = "/home/vellard/playlist_continuation/clustering-no-split/clean/200/"
    #filter_clusters_by_exact_match(clusters_dir, output_dir, threshold=2)
    input_file = "/content/drive/MyDrive/playlist_project/clustering-no-split/clusters/200/clusters_with_exact_matches.csv"
    output_dir = "/content/drive/MyDrive/playlist_project/clustering-no-split/clean/200/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "clusters_with_exact_matches.csv")
    clean_clusters(input_file, output_file, threshold=2)
    print("Cleaned clusters saved to:", output_file)

if __name__ == "__main__":
    main()