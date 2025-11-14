from file_utils import FileManager

def main():
    fm = FileManager()                
    df = fm.get_csv("/Users/intankwardhani/Documents/BeCode/immo-eliza-team-horses-analysis/data/properties_data_original.csv")
    print(df)

if __name__ == "__main__":
    main()

