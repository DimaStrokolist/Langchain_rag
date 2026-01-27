import telebot
from aiohttp.log import access_logger
from telebot import types
import  json
import requests
import uuid
import time
import asyncio
import telebot.async_telebot as async_telebot

from Langchain_example import agent_invoke, run_agent_and_get

token_tele = '8330630691:AAEtInbmX7Lxdm7cPPCPxFMekMyviYsVaDw'
bot = telebot.TeleBot(token_tele)
token_cache = {
    'token':None,
    'timestamp':0
}

@bot.message_handler(commands=['start'])
def start(message):

    bot.send_message(message.chat.id, "Введите операцию: ")
    bot.register_next_step_handler(message, langchain_my)

def langchain_my(message):
    result = run_agent_and_get(message.text)

    bot.send_message(message.chat.id, result)
bot.polling(none_stop=True)
