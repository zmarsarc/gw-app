from typing import List

import redis
from textual.app import App
from textual.containers import HorizontalGroup
from textual.reactive import reactive
from textual.widgets import Button, Label, ListItem, ListView, Static


class RedisKeyListView(ListView):

    keys = reactive([])

    def watch_keys(self, keys: List[str]):
        self.clear()

        if not keys:
            item = ListItem(Label("no items"), disabled=False)
            item.styles.height = "1"
            self.append(item)
            return

        for k in keys:
            item = ListItem(Label(k.decode()))
            item.styles.height = "1"
            self.append(item)


class StatusBar(HorizontalGroup):
    redis_connection = reactive(False)

    def compose(self):
        yield Static("redis disconnect.")

    def watch_redis_connection(self, is_connect: bool):
        if is_connect:
            label: Static = self.query_one(Static)
            label.update("redis connected.")


class InspectorApp(App):

    redis_connection = reactive(None)

    def __init__(self, driver_class=None, css_path=None, watch_css=False, ansi_color=False):
        super().__init__(driver_class, css_path, watch_css, ansi_color)
        self.styles.layout = "vertical"

    def compose(self):

        btn_connect = Button(
            "Connect Redis", id="btn_connect_redis", variant="success")
        yield btn_connect

        rlv = RedisKeyListView()
        rlv.styles.width = "1fr"
        view = Static("no item selected")
        view.styles.width = "1fr"
        yield HorizontalGroup(rlv, view)

        statusbar = StatusBar(classes="statusbar")
        statusbar.styles.dock = "bottom"
        yield statusbar

    def on_button_pressed(self, evt: Button.Pressed):
        if evt.button.id == "btn_connect_redis":
            rdb = redis.Redis("127.0.0.1")
            self.redis_connection = rdb

    def watch_redis_connection(self, rdb: redis.Redis):
        if rdb is None:
            return
        if rdb.ping():
            sb: StatusBar = self.query_one(".statusbar")
            sb.redis_connection = True

            keys = rdb.keys()
            self.query_one(RedisKeyListView).keys = keys


if __name__ == "__main__":
    app = InspectorApp()
    app.run()

