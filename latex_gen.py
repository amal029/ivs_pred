import shutil
import pandas as pd
import numpy as np

def dm_rmse_r2_best(models):
    # XXX: Generate latex tables for best dmtest rmse and r2 
    for otype in ['put']:
        for ts in [5]:
            for dd in ['figs']:
                df = pd.read_csv('./plots/dstat_%s_%s_%s_best_rmse.csv' % (
                    otype, ts, dd))
                dfp = pd.read_csv('./plots/pval_%s_%s_%s_best_rmse.csv' % (
                    otype, ts, dd))
                
                dfr = pd.read_csv('./plots/r2cmp_%s_%s_%s_best.csv' % (
                    otype, ts, dd))
                dfrp = pd.read_csv('./plots/r2cmp_pval_%s_%s_%s_best.csv' % (
                    otype, ts, dd))
                
                
                # rename the column headers
                columns = ['Point-SAM', 'Skew-HAR', '\\ac{TS}-SAM', 'Surface-HAR']
                index = ['Point-SAM', 'Skew-HAR', '\\ac{TS}-SAM', 'Surface-HAR']

                df.columns = columns
                df.index = index
                dfp.columns = columns
                dfp.index = index

                dfr.columns = columns
                dfr.index = index
                dfrp.columns = columns
                dfrp.index = index

                # Remove the surface index
                df = df.drop('Surface-HAR', axis=0)
                dfp = dfp.drop('Surface-HAR', axis=0)

                dfr = dfr.drop('Surface-HAR', axis=0)
                dfrp = dfrp.drop('Surface-HAR', axis=0)

                # set the values past the diagonal to nan
                for i in range(df.shape[0]):
                    for j in range(df.shape[1]):
                        if i >= j:
                            df.iloc[i, j] = np.nan
                            dfr.iloc[i, j] = np.nan

                # Remove the Point column
                df = df.drop('Point-SAM', axis=1)
                dfp = dfp.drop('Point-SAM', axis=1)

                dfr = dfr.drop('Point-SAM', axis=1)
                dfrp = dfrp.drop('Point-SAM', axis=1)

                # Map function to check if the corrisponding p-value is significant
                # Bold insignificant values
                def check_significance(x, xp):
                    result = np.where(xp > 0.05 ,f"font-weight: bold;" , None)

                    # if result.all():
                    #     # set all values to None
                    #     result = np.where(xp < 0.05, None, None)

                    # # Set value to None if x is nan 
                    # result = np.where(x != x, None, result)
                    
                    return result
                
                s = df.style.apply(check_significance, xp=dfp, axis=None)
                sr = dfr.style.apply(check_significance, xp=dfrp, axis=None)

                s.format('{:.1f}', na_rep=" ")
                sr.format('{:.1f}', na_rep=" ")

                # print(s.to_latex(column_format='l'*df.columns.size+'l', hrules=True, convert_css=True))
                # print(sr.to_latex(column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True))

                s.to_latex(buf='../feature_paper/figs/best/dstat-%s-%s-%s-best-rmse.tex' % (
                    otype, ts, dd), column_format='l'*df.columns.size+'l', hrules=True, convert_css=True) 
                sr.to_latex(buf='../feature_paper/figs/best/r2cmp-%s-%s-%s-best.tex' % (
                    otype, ts, dd), column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True)

                for i in range(len(models['best'])):
                    for j in range(i+1, len(models['best'])):
                        for stat in ['r2_stat', 'rmse_dstat']:
                            df = pd.read_csv('./plots/mt_%s_%s_%s_%s_%s_%s.csv' % (
                                stat, models['best'][i], models['best'][j], otype, ts, dd))
                            
                            dfp = pd.read_csv('./plots/mt_%s_pval_%s_%s_%s_%s_%s.csv' % (
                                stat.split('_')[0], models['best'][i], models['best'][j], otype, ts, dd))

                            # rename the column headers
                            columns = ['Index', 'm', 'Short-Term', 'Medium-Term', 'Long-Term']
                            df.columns = columns
                            dfp.columns = columns
                            df['m'] = ['\\ac{ITM}', '\\ac{ATM}', '\\ac{OTM}']
                            dfp['m'] = ['\\ac{ITM}', '\\ac{ATM}', '\\ac{OTM}']
                            
                            # Remove column Index
                            df = df.drop('Index', axis=1)
                            df.set_index('m', inplace=True)
                            dfp = dfp.drop('Index', axis=1)
                            dfp.set_index('m', inplace=True)
                            print(df)

                            # Map function to check if the corrisponding p-value is significant
                            def check_significance(x, xp):
                                result = np.where(xp > 0.05 ,f"font-weight: bold;" , None)

                                # if result.all():
                                #     # set all values to None
                                #     result = np.where(xp < 0.05, None, None)

                                # # Set value to None if x is nan 
                                # result = np.where(x != x, None, result)
                                
                                return result
                            
                            s = df.style.apply(check_significance, xp=dfp, axis=None)
                            s.format('{:.2f}' )
                            
                            s.to_latex(buf='../feature_paper/figs/mt/mt-%s-%s-%s-%s-%s-%s-%s.tex' % (
                                stat, models['best'][i], models['best'][j], otype, ts, dd, 'best'), column_format='l'*df.columns.size+'l', hrules=True, convert_css=True)


if __name__ == '__main__':
    # Copying over the timeseries and stacked bar graph
    shutil.copyfile('./plots/call_figs_r2_time_series_best_models_5.pdf', '../feature_paper/figs/call_figs_r2_time_series_best_models_5.pdf')
    shutil.copyfile('./plots/call_figs_rmse_time_series_best_models_5.pdf', '../feature_paper/figs/call_figs_rmse_time_series_best_models_5.pdf')

    # # XXX: Generate latex table for best absolute r2 and rmse results
    # # Copy over best absolute r2 and rmse graphs
    # for otype in ['put']:
    #     shutil.copyfile('./plots/%s_figs_r2_avg_models.pdf' % (otype), '../feature_paper/figs/best_abs_r2_%s.pdf' % (otype))
    #     shutil.copyfile('./plots/%s_figs_r2std_avg_models.pdf' % (otype), '../feature_paper/figs/best_abs_r2std_%s.pdf' % (otype))
    #     shutil.copyfile('./plots/%s_figs_rmse_avg_models.pdf' % (otype), '../feature_paper/figs/best_abs_rmse_%s.pdf' % (otype))
    #     shutil.copyfile('./plots/%s_figs_rmsestd_avg_models.pdf' % (otype), '../feature_paper/figs/best_abs_rmsestd_%s.pdf' % (otype))

    # # df = pd.read_csv('./plots/call_figs_rmse_r2_avg_std_models_5.csv')

    # # columns = ['Index', 'Models', 'RMSE', 'RMSE STD', '$R^2$', '$R^2$ STD']
    # # df.columns = columns

    # # df.drop('Index', axis=1, inplace=True)

    # # df['Models'] = ['Point', 'Skew', 'Term Structure', 'Surface']

    # # df.set_index('Models', inplace=True)
# # # Bold the max r2 and min rmse
    # # def highlight_max(s):
    # #     is_max = s == s.max()
    # #     return ['font-weight: bold' if v else None for v in is_max]
    
    # # def highlight_min(s):
    # #     is_min = s == s.min()
    # #     return ['font-weight: bold' if v else None for v in is_min]
    
    # # s = df.style.apply(highlight_max, subset=['$R^2$'])
    # # s = s.apply(highlight_min, subset=['RMSE'])

    # # s.to_latex(buf='../feature_paper/figs/best/rmse_r2_avg_std_models_5.tex', column_format='l'*df.columns.size+'l', hrules=True, convert_css=True)

    # dm_rmse_r2_best(model)


    # # # XXX: Generate latex table for dmtest rmse
    # for otype in ['call']:
    #     for ts in [5, 10, 20]:
    #         for dd in ['figs']:
    #             for feature in ['surf', 'point', 'skew', 'termstructure']:
    #                 for mmodel in ['ridge', 'lasso', 'enet']:
    #                     df = pd.read_csv('./plots/dstat_%s_%s_%s_%s_%s_rmse.csv' % (
    #                         otype, ts, dd, feature, mmodel))
    #                     dfp = pd.read_csv('./plots/pval_%s_%s_%s_%s_%s_rmse.csv' % (
    #                         otype, ts, dd, feature, mmodel))
                        
    #                     dfr = pd.read_csv('./plots/r2cmp_%s_%s_%s_%s_%s.csv' % (
    #                         otype, ts, dd, feature, mmodel))
    #                     dfrp = pd.read_csv('./plots/r2cmp_pval_%s_%s_%s_%s_%s.csv' % (
    #                         otype, ts, dd, feature, mmodel))
                        
                        
    #                     # rename the column headers
    #                     columns = ['SAM', '\\ac{PCA}', '\\ac{CCA}', '\\ac{NS}', '\\ac{ADNS}', '\\ac{SSVI}', '\\ac{HAR}', '\\ac{VAE}']
    #                     index = ['SAM', '\\ac{PCA}', '\\ac{CCA}', '\\ac{NS}', '\\ac{ADNS}', '\\ac{SSVI}', '\\ac{HAR}', '\\ac{VAE}']

    #                     if feature == 'surf':
    #                         columns.remove('\\ac{NS}')
    #                         index.remove('\\ac{NS}')
    #                     elif feature == 'point':
    #                         columns.remove('\\ac{SSVI}')
    #                         columns.remove('\\ac{ADNS}')
    #                         columns.remove('\\ac{NS}')
    #                         columns.remove('\\ac{VAE}')
    #                         index.remove('\\ac{SSVI}')
    #                         index.remove('\\ac{ADNS}')
    #                         index.remove('\\ac{NS}')
    #                         index.remove('\\ac{VAE}')

    #                     else:
    #                         columns.remove('\\ac{SSVI}')
    #                         columns.remove('\\ac{ADNS}')
    #                         index.remove('\\ac{SSVI}')
    #                         index.remove('\\ac{ADNS}')


    #                     df.columns = columns
    #                     df.index = index
    #                     dfp.columns = columns
    #                     dfp.index = index

    #                     dfr.columns = columns
    #                     dfr.index = index
    #                     dfrp.columns = columns
    #                     dfrp.index = index

    #                     # Remove the last index 
    #                     if feature == 'point':
    #                         df = df.drop('\\ac{HAR}', axis=0)
    #                         dfp = dfp.drop('\\ac{HAR}', axis=0)

    #                         dfr = dfr.drop('\\ac{HAR}', axis=0)
    #                         dfrp = dfrp.drop('\\ac{HAR}', axis=0)
    #                     else:
    #                         df = df.drop('\\ac{VAE}', axis=0)
    #                         dfp = dfp.drop('\\ac{VAE}', axis=0)

    #                         dfr = dfr.drop('\\ac{VAE}', axis=0)
    #                         dfrp = dfrp.drop('\\ac{VAE}', axis=0)


    #                     # remove the har index if ts != 20
    #                     # if ts != 20:
    #                     #     dfp = dfp.drop('\\ac{HAR}', axis=0)
    #                     #     df = df.drop('\\ac{HAR}', axis=0)

    #                     #     dfrp = dfrp.drop('\\ac{HAR}', axis=0)
    #                     #     dfr = dfr.drop('\\ac{HAR}', axis=0)

    #                     #     #convert har to nan
    #                     #     df['\\ac{HAR}'] = np.nan
    #                     #     dfr['\\ac{HAR}'] = np.nan

    #                     # set the values past the diagonal to nan
    #                     for i in range(df.shape[0]):
    #                         for j in range(df.shape[1]):
    #                             if i >= j:
    #                                 df.iloc[i, j] = np.nan
    #                                 dfr.iloc[i, j] = np.nan

    #                     # Drop the first column
    #                     df = df.drop('SAM', axis=1)
    #                     dfp = dfp.drop('SAM', axis=1)

    #                     dfr = dfr.drop('SAM', axis=1)
    #                     dfrp = dfrp.drop('SAM', axis=1)

    #                     # Map function to check if the corrisponding p-value is significant
    #                     def check_significance(x, xp):
    #                         result = np.where(xp > 0.05 ,f"font-weight: bold;" , None)

    #                         # if result.all():
    #                         #     # set all values to None
    #                         #     result = np.where(xp < 0.05, None, None)

    #                         # # Set value to None if x is nan 
    #                         # result = np.where(x != x, None, result)
                            
    #                         return result
                        
    #                     s = df.style.apply(check_significance, xp=dfp, axis=None)
    #                     sr = dfr.style.apply(check_significance, xp=dfrp, axis=None)
                        
    #                     # only style the dstat values if they are significant
    #                     # for i in range(df.shape[0]):
    #                     #     for j in range(df.shape[1]):
    #                     #         if dfp.iloc[i, j] > 0.05:
    #                     #             df.iloc[i, j] = np.nan

    #                     s.format('{:.1f}', na_rep=" ")
    #                     sr.format('{:.1f}', na_rep=" ")

    #                     # print(s.to_latex(column_format='l'*df.columns.size+'l', hrules=True, convert_css=True))
    #                     # print(sr.to_latex(column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True))

    #                     s.to_latex(buf='../feature_paper/figs/dstat/dstat-%s-%s-%s-%s-%s-rmse.tex' % (
    #                         otype, ts, dd, feature, mmodel), column_format='l'*df.columns.size+'l', hrules=True, convert_css=True) 
    #                     sr.to_latex(buf='../feature_paper/figs/r2cmp/r2cmp-%s-%s-%s-%s-%s.tex' % (
    #                         otype, ts, dd, feature, mmodel), column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True)

    # #                     # df.to_latex(buf='./plots/%s_%s_%s_%s_rmse.tex' % (
    # #                     #     otype, ts, dd, mmodel), header=True)

    # for otype in ['call']:
    #         for ts in [5, 10, 20]:
    #             for dd in ['figs']:
    #                 for feature in ['surf', 'point', 'skew', 'termstructure']:
    #                     mmodel = 'all'
    #                     df = pd.read_csv('./plots/dstat_%s_%s_%s_%s_%s_rmse.csv' % (
    #                             otype, ts, dd, feature, mmodel))
    #                     dfp = pd.read_csv('./plots/pval_%s_%s_%s_%s_%s_rmse.csv' % (
    #                         otype, ts, dd, feature, mmodel))
                        
    #                     dfr = pd.read_csv('./plots/r2cmp_%s_%s_%s_%s_%s.csv' % (
    #                         otype, ts, dd, feature, mmodel))
    #                     dfrp = pd.read_csv('./plots/r2cmp_pval_%s_%s_%s_%s_%s.csv' % (
    #                         otype, ts, dd, feature, mmodel))

    #                     if feature == 'surf' or feature ==  'skew':
    #                         ridge_col = 'HAR'
    #                     else: 
    #                         ridge_col = 'SAM'
                        
    #                     # rename the column headers
    #                     columns = ['Ridge-%s' % ridge_col, 'Lasso-\\ac{CCA}', 'Elastic Net-\\ac{CCA}']
    #                     index = ['Ridge-%s' % ridge_col, 'Lasso-\\ac{CCA}', 'Elastic Net-\\ac{CCA}']

    #                     df.columns = columns
    #                     df.index = index
    #                     dfp.columns = columns
    #                     dfp.index = index

    #                     dfr.columns = columns
    #                     dfr.index = index
    #                     dfrp.columns = columns
    #                     dfrp.index = index

    #                     # Remove the elastic net cca index
    #                     df = df.drop('Elastic Net-\\ac{CCA}', axis=0)
    #                     dfp = dfp.drop('Elastic Net-\\ac{CCA}', axis=0)

    #                     dfr = dfr.drop('Elastic Net-\\ac{CCA}', axis=0)
    #                     dfrp = dfrp.drop('Elastic Net-\\ac{CCA}', axis=0)


    #                     # set the values past the diagonal to nan
    #                     for i in range(df.shape[0]):
    #                         for j in range(df.shape[1]):
    #                             if i >= j:
    #                                 df.iloc[i, j] = np.nan
    #                                 dfr.iloc[i, j] = np.nan

    #                     # Drop the first column
    #                     df = df.drop('Ridge-%s' % ridge_col, axis=1)
    #                     dfp = dfp.drop('Ridge-%s' % ridge_col, axis=1)

    #                     dfr = dfr.drop('Ridge-%s' % ridge_col, axis=1)
    #                     dfrp = dfrp.drop('Ridge-%s' % ridge_col, axis=1)


    #                     # Map function to check if the corrisponding p-value is significant
    #                     def check_significance(x, xp):
    #                         result = np.where(xp > 0.05 ,f"font-weight: bold;" , None)

    #                         # if result.all():
    #                         #     # set all values to None
    #                         #     result = np.where(xp < 0.05, None, None)

    #                         # # Set value to None if x is nan 
    #                         # result = np.where(x != x, None, result)
                            
    #                         return result
                        
    #                     s = df.style.apply(check_significance, xp=dfp, axis=None)
    #                     sr = dfr.style.apply(check_significance, xp=dfrp, axis=None)
                        
    #                     s.format('{:.1f}', na_rep=" ")
    #                     sr.format('{:.1f}', na_rep=" ")

    #                     # print(s.to_latex(column_format='l'*df.columns.size+'l', hrules=True, convert_css=True))
    #                     # print(sr.to_latex(column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True))

    #                     s.to_latex(buf='../feature_paper/figs/dstat/dstat-%s-%s-%s-%s-%s-rmse.tex' % (
    #                         otype, ts, dd, feature, mmodel), column_format='l'*df.columns.size+'l', hrules=True, convert_css=True) 
    #                     sr.to_latex(buf='../feature_paper/figs/r2cmp/r2cmp-%s-%s-%s-%s-%s.tex' % (
    #                         otype, ts, dd, feature, mmodel), column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True)

    #                     # df.to_latex(buf='./plots/%s_%s_%s_%s_rmse.tex' % (
    #                     #     otype, ts, dd, mmodel), header=True) 

    # # # XXX: Generate latex table for dmtest and rmse cmp acrross lags
    # for otype in ['call']:
    #     for dd in ['figs']:
    #         for models in ['pmridge', 'tskridge']:
    #             df = pd.read_csv('./plots/lag_dstat_%s_%s_%s.csv' % (
    #                     otype, dd, models))
    #             dfp = pd.read_csv('./plots/lag_dstat_pval_%s_%s_%s.csv' % (
    #                 otype, dd, models))

    #             dfr = pd.read_csv('./plots/lag_r2_%s_%s_%s.csv' % (
    #                 otype, dd, models))
    #             dfrp = pd.read_csv('./plots/lag_r2_pval_%s_%s_%s.csv' % (
    #                 otype, dd, models))

    #             # rename the column headers
    #             columns = ['Lag 20', 'Lag 10', 'Lag 5']
    #             index = ['Lag 20', 'Lag 10', 'Lag 5']

    #             df.columns = columns
    #             df.index = index
    #             dfp.columns = columns
    #             dfp.index = index

    #             dfr.columns = columns
    #             dfr.index = index
    #             dfrp.columns = columns
    #             dfrp.index = index

    #             # Remove lag 5 index
    #             df = df.drop('Lag 5', axis=0)
    #             dfp = dfp.drop('Lag 5', axis=0)

    #             dfr = dfr.drop('Lag 5', axis=0)
    #             dfrp = dfrp.drop('Lag 5', axis=0)


    #             # set the values past the diagonal to nan
    #             for i in range(df.shape[0]):
    #                 for j in range(df.shape[1]):
    #                     if i >= j:
    #                         df.iloc[i, j] = np.nan
    #                         dfr.iloc[i, j] = np.nan

    #             # Remove the lag 20 column
    #             df = df.drop('Lag 20', axis=1)
    #             dfp = dfp.drop('Lag 20', axis=1)

    #             dfr = dfr.drop('Lag 20', axis=1)
    #             dfrp = dfrp.drop('Lag 20', axis=1)

    #             # Map function to check if the corrisponding p-value is significant
    #             def check_significance(x, xp):
    #                 result = np.where(xp > 0.05 ,f"font-weight: bold;" , None)

    #                 # if result.all():
    #                 #     # set all values to None
    #                 #     result = np.where(xp < 0.05, None, None)

    #                 # # Set value to None if x is nan 
    #                 # result = np.where(x != x, None, result)
    
    #                 return result

    #             s = df.style.apply(check_significance, xp=dfp, axis=None)
    #             sr = dfr.style.apply(check_significance, xp=dfrp, axis=None)

    #             s.format('{:.1f}', na_rep=" ")
    #             sr.format('{:.1f}', na_rep=" ")

    #             # print(s.to_latex(column_format='l'*df.columns.size+'l', hrules=True, convert_css=True))
    #             # print(sr.to_latex(column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True))

    #             s.to_latex(buf='../feature_paper/figs/dstat/lag-dstat-%s-%s-%s-rmse.tex' % (
    #                 otype, dd, models), column_format='l'*df.columns.size+'l', hrules=True, convert_css=True) 
    #             sr.to_latex(buf='../feature_paper/figs/r2cmp/lag-r2cmp-%s-%s-%s.tex' % (
    #                 otype, dd, models), column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True)

                # df.to_latex(buf='./plots/%s_%s_%s_%s_rmse.tex' % (
                #     otype, ts, dd, mmodel), header=True)  

    # df.to_latex(buf='./plots/%s_%s_%s_%s_rmse.tex' % (
    #     otype, ts, dd, mmodel), header=True)

    pass