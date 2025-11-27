import logging
import asyncio
from data.config import BOT_TOKEN

from aiogram import F
from aiogram import Bot, Dispatcher
from aiogram.types import Message, BotCommand
from aiogram.filters import CommandStart

from src.core.chain import SupportRAG

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
rag = SupportRAG()

async def set_commands(bot: Bot):
    commands = [
        BotCommand(command="start", description="Запустить бота поддержки"),
    ]
    await bot.set_my_commands(commands)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я бот поддержки. Задай мне вопрос по правилам, и я постараюсь помочь!"
    )

@dp.message(F.text)
async def handle_message(message: Message):
    user_question = message.text
    bot_msg = await message.answer("🔍 Ищу ответ...") 
    answer = await rag.process_question(user_question)
    await bot_msg.edit_text(answer) 


async def main():
    await set_commands(bot)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())