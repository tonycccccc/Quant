import sqlite3
conn = sqlite3.connect('trading.db')
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
print('Tables:', tables)
wl = conn.execute('SELECT ticker, company_name FROM watchlist ORDER BY ticker').fetchall()
print(f'Watchlist ({len(wl)} rows):', [r[0] for r in wl])
conn.close()
