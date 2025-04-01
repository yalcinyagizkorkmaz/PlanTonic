import os
import psycopg2
from dotenv import load_dotenv
import time
from psycopg2.extras import RealDictCursor
import traceback
from contextlib import contextmanager
import threading

print("env yüklendi")
load_dotenv()


class Database:
    def __init__(self):
        self.pool = None
        self._connection = None
        self._lock = threading.Lock()
        self._ping_timer = None
        self._stop_ping = False
        self._create_connection()
        
        self._setup_keepalive_ping()

    def _create_connection(self):
        """Veritabanı bağlantısı oluşturur - havuz yerine tek bağlantı kullanır"""
        with self._lock:
            try:
            
                if self._connection and not self._connection.closed:
                    try:
                        self._connection.close()
                    except Exception as e:
                        print(f"Bağlantı kapatma hatası: {e}")

                db_url = os.getenv("DATABASE_URL")
                if not db_url:
                    print("DATABASE_URL bulunamadı, varsayılan Neon bağlantısı kullanılıyor...")
                    db_url = "postgresql://plantonic_owner:npg_O7bYJqFeXvp1@ep-wild-cake-a2h9a5ec-pooler.eu-central-1.aws.neon.tech/plantonic"
                
         
                if "?" in db_url:
                    db_url = db_url.split("?")[0]  
                
             
                db_url += "?sslmode=require"
                
                print("Neon veritabanına bağlanılıyor...")
                print(f"Bağlantı URL'si: {db_url.replace('postgresql://plantonic_owner:npg_O7bYJqFeXvp1', 'postgresql://USER:PASSWORD')}")
                
                # SADECE temel parametrelerle bağlan - havuz KULLANMA
                self._connection = psycopg2.connect(
                    dsn=db_url,
                    cursor_factory=RealDictCursor
                )
                
                # Auto-commit modunu kapat
                self._connection.autocommit = False
                
                # Bağlantı kurduktan sonra timeout ayarlarını yap
                with self._connection.cursor() as cur:
                    cur.execute("SET statement_timeout = '30000';")  # 30 saniye
                    self._connection.commit()
                
                print("Veritabanı bağlantısı başarıyla oluşturuldu!")

            except Exception as e:
                print(f"Veritabanı bağlantı hatası: {e}")
                print("Hata detayı:")
                print(traceback.format_exc())
                time.sleep(3)
                try:
                    # Sadece DSN ile dene - daha basit parametrelerle
                    self._connection = psycopg2.connect(dsn=db_url)
                    print("İkinci denemede sadece DSN parametresiyle bağlantı oluşturuldu!")
                except Exception as e2:
                    print(f"İkinci bağlantı denemesi de başarısız: {e2}")
                    self._connection = None
                    raise

    def _setup_keepalive_ping(self):
        """Periyodik olarak bağlantıyı canlı tutmak için bir ping timer'ı ayarlar"""
        def ping_db():
            if self._stop_ping:
                return
                
            try:
                # Bağlantı kontrolü
                if not self._connection or self._connection.closed:
                    print("Bağlantı kapalı, yenileniyor...")
                    self._create_connection()
                
                # Her 15 saniyede bir basit bir sorgu çalıştır
                with self._connection.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    if result:
                        print("Ping başarılı - bağlantı aktif.")
                    else:
                        print("Ping sorgusu sonuç döndürmedi!")
                
                # Bağlantı aktif kalması için commit gerekebilir
                self._connection.commit()
                
            except Exception as e:
                print(f"Ping hatası: {e}")
                print("Hatadan sonra bağlantıyı yenileme deneniyor...")
                try:
                    self._create_connection()
                except:
                    print("Bağlantı yenileme başarısız!")
            
            # Yeni timer oluştur
            if not self._stop_ping:
                self._ping_timer = threading.Timer(15, ping_db)
                self._ping_timer.daemon = True  # Uygulamadan çıkışı engellemesin
                self._ping_timer.start()
        
        # İlk ping'i başlat
        self._ping_timer = threading.Timer(3, ping_db)
        self._ping_timer.daemon = True
        self._ping_timer.start()

    def _test_connection(self):
        """Bağlantının hala açık ve kullanılabilir olduğunu test eder"""
        try:
            if not self._connection or self._connection.closed:
                return False
                
            with self._connection.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result is not None
        except:
            return False

    @contextmanager
    def get_db_connection(self):
        """Bağlantı yönetimi - thread-safe tek bağlantı"""
        with self._lock:
            if not self._connection or self._connection.closed:
                self._create_connection()
                
            # Bağlantıyı test et
            if not self._test_connection():
                print("Bağlantı geçersiz, yenileniyor...")
                self._create_connection()
                
            try:
                yield self._connection
                self._connection.commit()
            except Exception as e:
                print(f"Bağlantı hatası: {e}")
                try:
                    self._connection.rollback()
                except:
                    pass
                
                # Kapanmış bir bağlantı mı?
                if isinstance(e, (psycopg2.OperationalError, psycopg2.InterfaceError)) or "closed" in str(e).lower():
                    print("Bağlantı yenileniyor...")
                    self._create_connection()
                
                raise

    def execute_query(self, query, params=None):
        """SQL sorgusu çalıştırır"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self.get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        return True
            except (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
                retry_count += 1
                print(f"Veritabanı sorgu hatası (Deneme {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(2)
                    if "closed" in str(e).lower() or "unexpected" in str(e).lower():
                        try:
                            self._create_connection()
                        except Exception as conn_error:
                            print(f"Bağlantı yenileme hatası: {conn_error}")
            except Exception as e:
                print(f"Beklenmeyen sorgu hatası: {e}")
                print(traceback.format_exc())
                return False
        
        print(f"Maksimum yeniden deneme sayısına ulaşıldı!")
        return False

    def fetch_all(self, query, params=None):
        """Tüm sonuçları getirir"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self.get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        result = cur.fetchall()
                        return result or []  # Boş sonuç durumunda boş liste döndür
            except (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
                retry_count += 1
                print(f"Veritabanı sorgu hatası (Deneme {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(2)
                    if "closed" in str(e).lower() or "unexpected" in str(e).lower():
                        try:
                            self._create_connection()
                        except Exception as conn_error:
                            print(f"Bağlantı yenileme hatası: {conn_error}")
            except Exception as e:
                print(f"Veri çekme hatası: {e}")
                print(traceback.format_exc())
                return []
        
        print("Maksimum yeniden deneme sayısına ulaşıldı!")
        return []

    def fetch_one(self, query, params=None):
        """Tek bir sonuç getirir"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self.get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        result = cur.fetchone()
                        return result
            except (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
                retry_count += 1
                print(f"Veritabanı sorgu hatası (Deneme {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(2)
                    if "closed" in str(e).lower() or "unexpected" in str(e).lower():
                        try:
                            self._create_connection()
                        except Exception as conn_error:
                            print(f"Bağlantı yenileme hatası: {conn_error}")
            except Exception as e:
                print(f"Veri çekme hatası: {e}")
                print(traceback.format_exc())
                return None
        
        print("Maksimum yeniden deneme sayısına ulaşıldı!")
        return None

    def keep_connection_alive(self):
        """Bağlantıyı canlı tutmak için basit bir sorgu çalıştırır"""
        try:
            if not self._connection or self._connection.closed:
                self._create_connection()
                
            with self._connection.cursor() as cur:
                cur.execute("SELECT 1")
                self._connection.commit()
                return True
        except Exception as e:
            print(f"Bağlantı canlı tutma hatası: {e}")
            try:
                self._create_connection()
            except:
                pass
            return False

    def disconnect(self):
        """Veritabanı bağlantısını kapatır"""
        # Timer'ı durdur
        self._stop_ping = True
        if self._ping_timer:
            try:
                self._ping_timer.cancel()
            except:
                pass
            
        with self._lock:
            if self._connection and not self._connection.closed:
                try:
                    self._connection.close()
                    self._connection = None
                    print("Veritabanı bağlantısı kapatıldı.")
                except Exception as e:
                    print(f"Bağlantı kapatma hatası: {e}")


def init_db(db):
    """Veritabanı ve tabloları oluşturur"""
    try:
        # Users tablosu
        db.execute_query("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Images tablosu
        db.execute_query("""
            CREATE TABLE IF NOT EXISTS images (
                image_id UUID PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Identification tablosu - sessionid unique kısıtlaması kaldırıldı
        db.execute_query("""
            CREATE TABLE IF NOT EXISTS identification (
                sessionid UUID,
                image_data TEXT NOT NULL,
                image_id UUID REFERENCES images(image_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (sessionid, image_id)
            )
        """)

        print("Veritabanı tabloları başarıyla oluşturuldu")
    except Exception as e:
        print(f"Veritabanı tabloları oluşturulurken hata: {e}")
        raise e


# Veritabanı örneğini oluştur
db = Database()

# Ayrı bir thread'de tabloları oluştur, böylece uygulama başlatması engellenmez
def initialize_database():
    try:
        time.sleep(3)  # Bağlantının oluşmasını bekle
        
        # Önce mevcut tabloları sil
        try:
            db.execute_query("DROP TABLE IF EXISTS identification CASCADE;")
            db.execute_query("DROP TABLE IF EXISTS images CASCADE;")
            db.execute_query("DROP TABLE IF EXISTS users CASCADE;")
            print("Eski tablolar başarıyla silindi")
        except Exception as e:
            print(f"Tablo silme hatası: {e}")
        
        # Tabloları yeniden oluştur
        init_db(db)
        print("Veritabanı tabloları başarıyla oluşturuldu")
            
    except Exception as e:
        print(f"Veritabanı başlatma hatası: {e}")

# Tabloları oluştur
db_init_thread = threading.Thread(target=initialize_database)
db_init_thread.daemon = True
db_init_thread.start()