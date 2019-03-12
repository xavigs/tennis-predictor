from selenium import webdriver
from selenium.webdriver.firefox.options import Options

SPORT = "Tenis"

driver = webdriver.Firefox(executable_path='..\geckodriver.exe')
driver.get('https://www.bet365.es/')
driver.find_element_by_link_text("Español").click()

# Go to Tennis markets
while True:
    try:
        left_menu_div = driver.find_elements_by_class_name("wn-WebNavModule")
        left_menu = left_menu_div[0].find_elements_by_class_name("wn-Classification")
        break
    except:
        pass

for link in left_menu:
    if link.text == SPORT:
        link.click()
        break

# Go to available games
while True:
    try:
        driver.find_elements_by_class_name("sm-CouponLink_Label")[0].click()
        break
    except:
        pass
