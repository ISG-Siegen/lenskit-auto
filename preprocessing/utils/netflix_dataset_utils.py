import os

# most code to read the data in is reused from https://gist.github.com/guimatheus92/5bf038f94abe46056b79a0a3a640e1bd


def read_netflix_data(filenames, base_path):
    # Open and create file for recording
    dataset = open(os.path.join(base_path, 'netflix/fullcombined_data.csv'), mode='w')
    # Create list for files rows
    rows = list()

    for filename in filenames:
        # Print a message
        print("Reading the file {}...".format(filename))
        # With the file open, we extract the rows
        with open(os.path.join(base_path, ('netflix/' + filename + '.txt'))) as f:
            # Loop through each row
            for row in f:
                # Deleting list content
                del rows[:]
                # Divide the row of the file by the end of line character
                row = row.strip()
                # If we find "colon" at the end of the row, we do replace by removing
                # the character, as we just want the movie id
                if row.endswith(':'):
                    movie_id = row.replace(':', '')
                    # If not, we create a comprehension list to separate the columns by comma
                else:
                    # Split the columns
                    rows = [x for x in row.split(',')]
                    # Use movie id at index zero position
                    rows.insert(0, movie_id)
                    # Write the result to the new file
                    dataset.write(','.join(rows))
                    dataset.write('\n')
        print("Finished.\n")
    dataset.close()
