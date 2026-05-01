import pandas as pd
import os

print("Starting HCRIS Parquet Compression...")

# The years we want to process
years = ["2022", "2023", "2024", "2025"]

# The exact column headers we need for the app
nmrc_cols = ['rpt_rec_num', 'wksht_cd', 'line_num', 'clmn_num', 'itm_val_num']
alpha_cols = ['rpt_rec_num', 'wksht_cd', 'line_num', 'clmn_num', 'itm_val_alphnmrc']
rpt_cols = ['rpt_rec_num', 'prvdr_ctrl_type_cd', 'prvdr_num'] + [f'v{i}' for i in range(3, 18)]

for year in years:
    print(f"\n--- Processing Fiscal Year {year} ---")
    
    # 1. Process ALPHA (Text Data)
    f_alpha = f"RNL11_{year}_alpha.csv"
    if os.path.exists(f_alpha):
        # We read as strings to prevent Pandas from dropping leading zeros
        df_alpha = pd.read_csv(f_alpha, header=None, names=alpha_cols, dtype=str)
        df_alpha.to_parquet(f"RNL11_{year}_alpha.parquet", index=False)
        print(f"✅ Converted {f_alpha}")
    else:
        print(f"⚠️ Missing {f_alpha}")

    # 2. Process NMRC (Numeric Data)
    f_nmrc = f"RNL11_{year}_nmrc.csv"
    if os.path.exists(f_nmrc):
        df_nmrc = pd.read_csv(f_nmrc, header=None, names=nmrc_cols, dtype=str)
        df_nmrc.to_parquet(f"RNL11_{year}_nmrc.parquet", index=False)
        print(f"✅ Converted {f_nmrc}")
    else:
        print(f"⚠️ Missing {f_nmrc}")
        
    # 3. Process RPT (Report Metadata)
    f_rpt = f"RNL11_{year}_rpt.csv"
    if os.path.exists(f_rpt):
        # Optimization: We only actually use columns 0 and 2 in the dashboard!
        # Dropping the rest here shrinks the file even further.
        df_rpt = pd.read_csv(f_rpt, header=None, names=rpt_cols, usecols=['rpt_rec_num', 'prvdr_num'], dtype=str)
        df_rpt.to_parquet(f"RNL11_{year}_rpt.parquet", index=False)
        print(f"✅ Converted {f_rpt} (Optimized)")
    else:
        print(f"⚠️ Missing {f_rpt}")

print("\n🎉 All files successfully compressed to Parquet!")